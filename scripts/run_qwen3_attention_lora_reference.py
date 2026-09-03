"""Run a one-step Qwen3 q_proj/v_proj LoRA reference update."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2 or min(args.batch_size, args.rank) <= 0:
        raise ValueError("sequence length must be at least 2; batch size and rank must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.concurrent_vllm_lora import inject_lora, trainable_parameters

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.requires_grad_(False)
    injected = inject_lora(model, rank=args.rank, alpha=args.alpha)
    if not injected:
        raise RuntimeError("no q_proj/v_proj modules were found")
    model.to("cuda")
    model.train()
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    seed = (
        "高吞吐推理需要连续批处理，参数高效微调只更新低秩适配器。"
        " A shared base model serves inference while a LoRA adapter is trained. "
    )
    token_ids = tokenizer.encode(seed, add_special_tokens=False)
    while len(token_ids) < args.sequence_length:
        token_ids.extend(token_ids)
    sample = torch.tensor(token_ids[: args.sequence_length], dtype=torch.long, device="cuda")
    inputs = sample.unsqueeze(0).repeat(args.batch_size, 1)
    probe = inputs[:1].clone()
    probe[0, 0] = (int(probe[0, 0]) + 1) % int(model.config.vocab_size)

    with torch.no_grad():
        before_logits = model(input_ids=probe, use_cache=False).logits[:, -1, :].float()

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=inputs, labels=inputs, use_cache=False)
    loss = output.loss
    loss.backward()
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        after_logits = model(input_ids=probe, use_cache=False).logits[:, -1, :].float()
    logit_delta = (after_logits - before_logits).abs()
    result = {
        "success": bool(torch.isfinite(loss)) and gradients_finite and bool(torch.isfinite(after_logits).all()),
        "model": str(model_path),
        "target_modules": ["q_proj", "v_proj"],
        "injected_module_count": len(injected),
        "injected_module_examples": list(injected[:4]),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "rank": args.rank,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_fraction": sum(parameter.numel() for parameter in parameters)
        / sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "gradients_finite": gradients_finite,
        "max_abs_logit_delta": float(logit_delta.max().detach().cpu()),
        "mean_abs_logit_delta": float(logit_delta.mean().detach().cpu()),
        "step_elapsed_s": elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
