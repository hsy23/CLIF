
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
import torch
import os
import json
import random



def parse_replica_gpus(args):
    if args.replica_gpus:
        try:
            replica_gpu_list = json.loads(args.replica_gpus)
            print(f"[model_loader] Manual GPU assignment: {replica_gpu_list}")
            return replica_gpu_list
        except json.JSONDecodeError as e:
            print(f"[model_loader] Failed to parse replica_gpus; falling back to round-robin assignment: {e}")

    total_gpus = torch.cuda.device_count()
    num_replicas = args.num_replicas
    print(f"[model_loader] Available GPUs: {total_gpus}; replicas: {num_replicas}")

    replica_gpu_list = [[i % total_gpus] for i in range(num_replicas)]
    print(f"[model_loader] Default GPU assignment: {replica_gpu_list}")
    return replica_gpu_list



def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer



def _resolve_lora_target(model_name):
    name = model_name.lower()
    if "gpt2" in name:
        return ["c_attn", "c_proj"], False
    if "llama" in name or "mistral" in name:
        return ["q_proj", "v_proj"], False
    if "qwen" in name:
        if "qwen3" in name or "qwen-3" in name:
            return ["q_proj", "v_proj", "k_proj", "o_proj"], False
        return ["c_attn", "c_proj"], True
    if "bert" in name:
        return ["query", "value"], False
    print(f"[model_loader] Unknown model family for {model_name}; using fallback LoRA targets")
    return ["query_key_value"], False


def _make_lora_config(model_name, low_rank):
    target_modules, fan_in_fan_out = _resolve_lora_target(model_name)
    return LoraConfig(
        r=low_rank, lora_alpha=32, lora_dropout=0.1,
        bias="none", target_modules=target_modules,
        fan_in_fan_out=fan_in_fan_out,
    )


def _finalize_model(model, device, quantization):
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    if quantization is None and device is not None:
        model = model.to(device)
    return model



def _resolve_quantization(quantization, force_quantization):
    if force_quantization != "none":
        print(f"[model_loader] Forced quantization: {force_quantization}")
        return force_quantization
    if quantization is not None:
        return quantization
    print("[model_loader] No quantization selected; using FP16")
    return None


def _load_base_model(model_name, quantization, device_map):
    if quantization == "4bit":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=cfg, device_map=device_map)
    if quantization == "8bit":
        cfg = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16,
                                 bnb_8bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=cfg, device_map=device_map)
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)



def load_model_instance(model_name, low_rank, device, quantization=None, force_quantization="none"):
    quantization = _resolve_quantization(quantization, force_quantization)
    model = _load_base_model(model_name, quantization, device)

    lora_config = _make_lora_config(model_name, low_rank)
    model.add_adapter(lora_config)
    model.enable_adapters()

    return _finalize_model(model, device, quantization)


def load_model_instance_multi_GPU(model_name, low_rank, device=None, device_map=None, force_quantization="none"):
    quantization = None
    if force_quantization != "none":
        quantization = force_quantization
        print(f"[model_loader] Multi-GPU load with quantization: {quantization}")

    if quantization is not None:
        print("[model_loader] Quantized multi-GPU loading uses the first assigned GPU")
        if isinstance(device_map, list) and device_map:
            device = torch.device(f"cuda:{device_map[0]}")
        return load_model_instance(model_name, low_rank, device, quantization, force_quantization)

    if isinstance(device_map, list):
        from accelerate import infer_auto_device_map
        from transformers.modeling_utils import get_max_memory
        max_memory = get_max_memory()
        filtered = {i: max_memory[i] for i in device_map if i in max_memory}
        filtered["cpu"] = max_memory["cpu"]
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="balanced", max_memory=filtered,
        )
    elif device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device,
        )

    lora_config = _make_lora_config(model_name, low_rank)
    model.add_adapter(lora_config)
    model.enable_adapters()

    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    if device_map is None and device is not None and not isinstance(device, dict):
        model = model.to(device)
    return model


def load_dual_model_instance(model_name, low_rank, device, active_adapter_name,
                             shadow_adapter_name, replica_id=0,
                             quantization=None, force_quantization="none"):
    quantization = _resolve_quantization(quantization, force_quantization)
    model = _load_base_model(model_name, quantization, device)

    lora_config = _make_lora_config(model_name, low_rank)
    model.add_adapter(lora_config, adapter_name=active_adapter_name)
    model.add_adapter(lora_config, adapter_name=shadow_adapter_name)
    model.set_adapter(active_adapter_name)

    model = prepare_model_for_kbit_training(model)
    if quantization is None:
        model = model.to(device)

    print(f"[model_loader] Dual adapters initialized on {device}")
    return model



def load_multi_token_datasets(train_dataset_names, train_size_ratio, infer_size_ratio,
                              num_replicas, tokenizer, max_length=128, seed=42,
                              max_train_samples=5000):
    def preprocess_function(examples):
        instructions = examples["instruction"]
        inputs_text = examples.get("input", [""] * len(instructions))
        if "response" in examples and "output" not in examples:
            examples["output"] = examples["response"]
        outputs = examples["output"]

        combined_text = [
            f"{inst}{tokenizer.eos_token}{inp}{tokenizer.eos_token}{out}"
            if inp and inp.strip() else f"{inst}{tokenizer.eos_token}{out}"
            for inst, inp, out in zip(instructions, inputs_text, outputs)
        ]

        inputs = tokenizer(combined_text, padding="max_length", truncation=True,
                           max_length=max_length, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        sep = tokenizer.eos_token_id
        for i in range(len(labels)):
            mask = inputs["attention_mask"][i]
            valid_tokens = labels[i] * mask
            sep_positions = (valid_tokens == sep).nonzero(as_tuple=True)[0]
            if len(sep_positions) >= 2:
                labels[i, : sep_positions[1] + 1] = -100
            elif len(sep_positions) == 1:
                labels[i, : sep_positions[0] + 1] = -100
        inputs["labels"] = labels
        return inputs

    raw_datasets = []
    raw_train_subsets_for_infer = []

    for i, train_name in enumerate(train_dataset_names):
        print(f"[data] Loading dataset {i}: {train_name}")
        try:
            ds = load_dataset(train_name, trust_remote_code=True)
            split_data = ds["train"]

            train_size = int(len(split_data) * train_size_ratio)
            infer_size = int(len(split_data) * infer_size_ratio)

            if train_size == 0:
                print(f"[data] Dataset {i} has an empty training split")
                raw_datasets.append(None)
                raw_train_subsets_for_infer.append(None)
                continue

            raw_datasets.append(split_data.select(range(train_size)))
            raw_train_subsets_for_infer.append(split_data.select(range(train_size, train_size + infer_size)))
            print(f"[data] Dataset {i} split sizes: train={train_size}, inference={infer_size}")
        except Exception as e:
            print(f"[data] Failed to load {train_name}: {e}")
            raw_datasets.append(None)
            raw_train_subsets_for_infer.append(None)

    valid_datasets = [ds for ds in raw_datasets if ds is not None]
    if not valid_datasets:
        print("[data] No valid datasets were loaded")
        return [None] * num_replicas, None

    tokenized_train_datasets = []
    used_indices = {}

    for i in range(num_replicas):
        dataset_idx = i % len(valid_datasets)
        actual_dataset_idx = 0
        valid_count = 0
        for j, ds in enumerate(raw_datasets):
            if ds is not None:
                if valid_count == dataset_idx:
                    actual_dataset_idx = j
                    break
                valid_count += 1

        current_dataset = raw_datasets[actual_dataset_idx]
        if actual_dataset_idx not in used_indices:
            used_indices[actual_dataset_idx] = set()

        available = list(set(range(len(current_dataset))) - used_indices[actual_dataset_idx])
        random.seed(seed + i)
        random.shuffle(available)

        samples_to_take = min(max_train_samples, len(available))
        if samples_to_take == 0:
            print(f"[data] No unused samples remain for dataset {actual_dataset_idx}; replica={i}")
            tokenized_train_datasets.append(None)
            continue

        selected = available[:samples_to_take]
        used_indices[actual_dataset_idx].update(selected)

        train_subset = current_dataset.select(selected)
        print(f"[data] Replica {i} <- dataset {actual_dataset_idx} ({train_dataset_names[actual_dataset_idx]}), samples={len(train_subset)}")

        tokenized_train = train_subset.map(
            preprocess_function, batched=True,
            remove_columns=train_subset.column_names,
            desc=f"Tokenizing replica {i}",
        )
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_train_datasets.append(tokenized_train)

    print("[data] Building mixed inference dataset")
    inference_samples_list = []
    valid_infer_count = sum(1 for ds in raw_train_subsets_for_infer if ds is not None)

    if valid_infer_count > 0:
        total_size = sum(len(ds) for ds in raw_train_subsets_for_infer if ds is not None)
        if total_size == 0:
            combined_raw_infer_dataset = None
        else:
            samples_per_source = min(total_size // valid_infer_count, 2000)
            for i, raw_subset in enumerate(raw_train_subsets_for_infer):
                if raw_subset is None:
                    continue
                actual = min(samples_per_source, len(raw_subset))
                if actual == 0:
                    continue
                print(f"[data] Inference samples from dataset {i} ({train_dataset_names[i]}): {actual}")
                shuffled = raw_subset.shuffle(seed=seed + i)
                inference_samples_list.append(shuffled.select(range(actual)))

            if not inference_samples_list:
                combined_raw_infer_dataset = None
            else:
                combined_raw_infer_dataset = concatenate_datasets(inference_samples_list).shuffle(seed=seed)
                print(f"[data] Mixed inference dataset size: {len(combined_raw_infer_dataset)}")
    else:
        combined_raw_infer_dataset = None

    print(f"[data] Prepared {len(tokenized_train_datasets)} training datasets and one mixed inference dataset")
    return tokenized_train_datasets, combined_raw_infer_dataset
