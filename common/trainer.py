import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class SimpleTrainingArguments:
    output_dir: str = field(default="./output")
    batch_size: int = field(default=8)
    learning_rate: float = field(default=2e-4)
    max_steps: int = field(default=100)
    gradient_accumulation_steps: int = field(default=8)
    logging_steps: int = field(default=20)
    save_steps: int = field(default=50)
    fp16: bool = field(default=True)
    max_grad_norm: float = field(default=1.0)
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)


class SimpleTrainer:
    def __init__(self, model, args=None, train_dataset=None, eval_dataset=None,
                 tokenizer=None, save_checkpoint=False):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        if args is None:
            args = SimpleTrainingArguments()

        self.args = args
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.max_steps = args.max_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.output_dir = args.output_dir
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.fp16 = args.fp16
        self.max_grad_norm = args.max_grad_norm
        self.weight_decay = args.weight_decay
        self.warmup_steps = args.warmup_steps
        self.save_cp = save_checkpoint

        os.makedirs(self.output_dir, exist_ok=True)

        self.global_step = 0
        self.state = type("obj", (object,), {"log_history": []})
        self.last_update_metrics = {}

        trainable_params = []
        fp32_trainable = 0
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                if param.dtype in (torch.float16, torch.bfloat16):
                    param.data = param.data.float()
                    fp32_trainable += param.numel()
                trainable_params.append(param)
            else:
                param.requires_grad = False
        if fp32_trainable > 0:
            print(f"[SimpleTrainer] Cast {fp32_trainable} trainable LoRA params to FP32 for stable AMP training")

        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        self.num_update_steps = self.max_steps

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_update_steps,
        )

        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 else None

        if train_dataset:
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
            )

        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset, batch_size=self.batch_size, shuffle=False,
            )

    def _prepare_inputs(self, batch):
        if isinstance(batch, dict) and "instruction" in batch and "output" in batch:
            instructions = batch["instruction"]
            inputs_text = batch.get("input", [""] * len(instructions))
            outputs = batch["output"]

            combined_text = [
                f"{inst}{self.tokenizer.eos_token}{inp}{self.tokenizer.eos_token}{out}"
                if inp and inp.strip() else f"{inst}{self.tokenizer.eos_token}{out}"
                for inst, inp, out in zip(instructions, inputs_text, outputs)
            ]

            inputs = self.tokenizer(
                combined_text, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            )

            labels = inputs["input_ids"].clone()
            sep = self.tokenizer.eos_token_id
            for i in range(len(labels)):
                mask = inputs["attention_mask"][i]
                valid_tokens = labels[i] * mask
                sep_positions = (valid_tokens == sep).nonzero(as_tuple=True)[0]
                if len(sep_positions) >= 2:
                    labels[i, : sep_positions[1] + 1] = -100
                elif len(sep_positions) == 1:
                    labels[i, : sep_positions[0] + 1] = -100

            inputs["labels"] = labels
        else:
            inputs = batch

        return {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def get_train_dataloader_iter(self):
        return iter(self.train_dataloader)

    def _count_batch_samples(self, inputs):
        for key in ("input_ids", "attention_mask", "labels"):
            value = inputs.get(key)
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
        return 0

    def _count_input_tokens(self, inputs):
        if "attention_mask" in inputs:
            return int(inputs["attention_mask"].sum().item())
        if "input_ids" in inputs:
            return int(inputs["input_ids"].numel())
        return 0

    def _count_target_tokens(self, inputs):
        if "labels" in inputs:
            return int((inputs["labels"] != -100).sum().item())
        return self._count_input_tokens(inputs)

    def train_one_update_step(self, dataloader_iter):
        self.model.train()
        total_loss = 0
        update_start_time = time.time()
        total_input_tokens = 0
        total_target_tokens = 0
        total_samples = 0
        for _ in range(self.gradient_accumulation_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                return None
            inputs = self._prepare_inputs(batch)
            total_input_tokens += self._count_input_tokens(inputs)
            total_target_tokens += self._count_target_tokens(inputs)
            total_samples += self._count_batch_samples(inputs)

            if self.fp16:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(**inputs)
                    loss = outputs.loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**inputs)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
            total_loss += loss.item() * self.gradient_accumulation_steps

        if self.fp16:
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm,
                )
            self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        update_end_time = time.time()
        update_wall_time = update_end_time - update_start_time
        self.last_update_metrics = {
            "update_start_time": update_start_time,
            "update_end_time": update_end_time,
            "update_wall_time_sec": update_wall_time,
            "update_input_tokens": total_input_tokens,
            "update_target_tokens": total_target_tokens,
            "update_samples": total_samples,
            "input_tokens_per_sec": total_input_tokens / update_wall_time if update_wall_time > 0 else 0.0,
            "target_tokens_per_sec": total_target_tokens / update_wall_time if update_wall_time > 0 else 0.0,
            "samples_per_sec": total_samples / update_wall_time if update_wall_time > 0 else 0.0,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "micro_batch_size": self.batch_size,
        }
        return total_loss / self.gradient_accumulation_steps

    def _save_checkpoint(self, step):
        output_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(output_dir, exist_ok=True)
        adapter_state = self.model.get_adapter_state_dict()
        torch.save(adapter_state, os.path.join(output_dir, "adapter_model.bin"))

        training_state = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "log_history": self.state.log_history,
        }
        if self.scaler:
            training_state["scaler_state"] = self.scaler.state_dict()
        torch.save(training_state, os.path.join(output_dir, "training_state.bin"))
        print(f"[SimpleTrainer] Saved checkpoint to {output_dir}")

    def load_checkpoint(self, checkpoint_dir):
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.bin")
        training_state_path = os.path.join(checkpoint_dir, "training_state.bin")

        if os.path.exists(adapter_path):
            adapter_state = torch.load(adapter_path, map_location="cuda:0")
            self.model.load_state_dict(adapter_state, strict=False)

        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cuda:0")
            self.global_step = training_state["global_step"]
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler_state"])
            self.state.log_history = training_state["log_history"]
            if self.scaler and "scaler_state" in training_state:
                self.scaler.load_state_dict(training_state["scaler_state"])

    def get_adapter_state_dict(self):
        return self.model.get_adapter_state_dict()
