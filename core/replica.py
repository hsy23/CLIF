
import time
import torch
import threading
from collections import deque
from common.trainer import SimpleTrainer, SimpleTrainingArguments
from common.state import StateManagement, ReplicaState
from common.evaluator import DialogueEvaluatorLite, CodeEvaluatorLite



class Replica:
    def __init__(self, replica_id, model, train_dataset, tokenizer, args,
                 state_manager=None, task_mode=None, quantization=None):
        self.replica_id = replica_id
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.train_batch_size = args.train_batch_size[replica_id]
        self.infer_batch_size = args.infer_batch_size[replica_id]
        self.infer_length = args.infer_length
        self.args = args

        self.performance_score = args.initial_score[replica_id]
        self.request_queue = deque()
        self.queue_lock = threading.Lock()
        self.serve_metrics = []
        self.training_metrics = []
        self.training_step_metrics = []
        self.is_idle = True
        self.state_manager = state_manager
        self.task_mode = task_mode
        self.quantization = quantization

        if self.task_mode == "conv":
            self.evaluator = DialogueEvaluatorLite(enable_bert=False)
        elif self.task_mode == "code":
            self.evaluator = CodeEvaluatorLite(enable_codebert=False)
        else:
            self.evaluator = None

        self.state_manager.register_replica(self.replica_id, ReplicaState.SERVING)

        self._recent_window_size = 3
        self._slo_recent = deque(maxlen=self._recent_window_size)
        self._wait_recent = deque(maxlen=self._recent_window_size)
        self._slo_hist_sum = 0.0
        self._slo_hist_cnt = 0
        self._wait_hist_sum = 0.0
        self._wait_hist_cnt = 0

        print(f"[Replica {self.replica_id}] initialized on device={self.model.device}")


    def _dequeue_batch(self):
        with self.queue_lock:
            if not self.request_queue:
                return []
            batch = []
            while self.request_queue and len(batch) < self.infer_batch_size:
                if self.request_queue[0]["timestamp"] + self.args.ddl <= time.time():
                    self.request_queue.popleft()
                    continue
                batch.append(self.request_queue.popleft())
        return batch

    def _build_prompts(self, batch):
        instructions = [req["instruction"] for req in batch]
        inputs_text = [req["input"] for req in batch]
        outputs_text = [req["output"] for req in batch]
        prompts = []
        for inst, inp in zip(instructions, inputs_text):
            if inp and inp.strip():
                prompts.append(f"{inst}{self.tokenizer.eos_token}{inp}")
            else:
                prompts.append(inst)
        return instructions, inputs_text, outputs_text, prompts

    def _tokenize_prompts(self, prompts, device=None):
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.infer_length,
        )
        dev = device or self.model.device
        return {k: v.to(dev) for k, v in inputs.items()}

    def _count_inference_input_tokens(self, inputs):
        if "attention_mask" in inputs:
            return int(inputs["attention_mask"].sum().item())
        if "input_ids" in inputs:
            return int(inputs["input_ids"].numel())
        return 0

    def _build_gen_kwargs(self, inputs):
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": self.args.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.generation_strategy == "sampling":
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.args.temperature
            gen_kwargs["top_p"] = self.args.top_p
        elif self.args.generation_strategy == "beam_search":
            gen_kwargs["num_beams"] = self.args.num_beams
            gen_kwargs["do_sample"] = False
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
        else:
            gen_kwargs["do_sample"] = False
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
        return gen_kwargs

    def _generate_texts(self, model, inputs, batch_size, **extra_gen_kwargs):
        generated_texts = []
        output_tokens = 0
        with torch.no_grad():
            try:
                gen_kwargs = self._build_gen_kwargs(inputs)
                gen_kwargs.update(extra_gen_kwargs)
                generated_ids = model.generate(**gen_kwargs)
                input_length = inputs["input_ids"].shape[1]
                for i in range(batch_size):
                    new_tokens = generated_ids[i, input_length:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    generated_texts.append(text if text else "[NO_CONTENT]")
                    output_tokens += len(new_tokens)
            except Exception as e:
                print(f"[Replica {self.replica_id}] generation failed: {e}")
                import traceback; traceback.print_exc()
                generated_texts = ["[ERROR]"] * batch_size
                output_tokens = 0
        return generated_texts, output_tokens

    def _compute_eval_loss(self, model, instructions, inputs_text, outputs_text,
                           device=None):
        loss = float("inf")
        perplexity = float("inf")
        try:
            full_texts = [
                f"{inst}{self.tokenizer.eos_token}{inp}{self.tokenizer.eos_token}{out}"
                if inp and inp.strip() else f"{inst}{self.tokenizer.eos_token}{out}"
                for inst, inp, out in zip(instructions, inputs_text, outputs_text)
            ]
            dev = device or model.device
            loss_inputs = self.tokenizer(
                full_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.infer_length,
            )
            loss_inputs = {k: v.to(dev) for k, v in loss_inputs.items()}
            labels = loss_inputs["input_ids"].clone()
            sep = self.tokenizer.eos_token_id
            for i in range(len(labels)):
                mask = loss_inputs["attention_mask"][i]
                valid_tokens = labels[i] * mask
                sep_positions = (valid_tokens == sep).nonzero(as_tuple=True)[0]
                if len(sep_positions) >= 2:
                    labels[i, : sep_positions[1] + 1] = -100
                elif len(sep_positions) == 1:
                    labels[i, : sep_positions[0] + 1] = -100
            loss_inputs["labels"] = labels
            with torch.no_grad():
                out = model(**loss_inputs)
            if out.loss is not None:
                loss = out.loss.item()
                perplexity = torch.exp(out.loss).item()
        except Exception as e:
            print(f"[Replica {self.replica_id}] evaluation loss failed: {e}")
        return loss, perplexity

    def _compute_slo_stats(self, batch, s_batch, output_tokens, start_time, finished_time):
        arrival_time = float("inf")
        success_tokens = 0
        avg_tokens = output_tokens / len(batch) if batch else 0
        for req in batch:
            req_time = req.get("timestamp", float("inf"))
            arrival_time = min(arrival_time, req_time)
            if finished_time - req_time <= req.get("ddl", float("inf")):
                success_tokens += int(avg_tokens)
        return arrival_time, success_tokens

    def _build_metrics_record(self, batch, s_batch, process_time, loss, perplexity,
                              arrival_time, finished_time, input_tokens, output_tokens, success_tokens):
        return {
            "replica_id": self.replica_id,
            "infer_batch_size": len(batch),
            "s_batch": s_batch,
            "set_batch": self.infer_batch_size,
            "process_time": process_time,
            "total_service_time": finished_time - arrival_time,
            "avg_time_per_request": process_time / len(batch),
            "eval_loss": loss,
            "eval_perplexity": perplexity,
            "arrival_time": arrival_time,
            "finished_time": finished_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_inference_tokens": input_tokens + output_tokens,
            "success_tokens": success_tokens,
        }

    def select_evaluator(self, generated_texts, reference_texts, task_mode,
                         instructions=None, language="python", **kwargs):
        def _sanitize(xs):
            return [x if isinstance(x, str) else ("" if x is None else str(x)) for x in (xs or [])]

        gen = _sanitize(generated_texts)
        ref = _sanitize(reference_texts)
        ins = None if instructions is None else _sanitize(instructions)

        if task_mode == "conv":
            eval_results = self.evaluator.evaluate_batch(gen, ref, ins, bert_lang=kwargs.get("bert_lang", "en"))
        elif task_mode == "code":
            eval_results = self.evaluator.evaluate_batch(generated_codes=gen, reference_codes=ref,
                                                          instructions=ins, language=language)
        else:
            return {}

        avg_metrics = {}
        if task_mode == "conv" and eval_results:
            if "bleu" in eval_results[0] and eval_results[0]["bleu"] is not None:
                vals = [r["bleu"] for r in eval_results if r.get("bleu") is not None]
                if vals:
                    avg_metrics["avg_bleu"] = sum(vals) / len(vals)
                    avg_metrics["sum_bleu"] = sum(vals)
            if "rouge" in eval_results[0] and eval_results[0]["rouge"] is not None:
                for key in ("rouge1", "rouge2", "rougeL"):
                    vals = [r["rouge"][key] for r in eval_results if r.get("rouge", {}).get(key) is not None]
                    if vals:
                        avg_metrics[f"avg_{key}"] = sum(vals) / len(vals)
                        avg_metrics[f"sum_{key}"] = sum(vals)
            if "bert" in eval_results[0] and eval_results[0]["bert"] is not None:
                vals = [r["bert"]["f1"] for r in eval_results if r.get("bert", {}).get("f1") is not None]
                if vals:
                    avg_metrics["avg_bert_f1"] = sum(vals) / len(vals)
                    avg_metrics["sum_bert_f1"] = sum(vals)
            if "coherence" in eval_results[0] and eval_results[0]["coherence"] is not None:
                vals = [r["coherence"] for r in eval_results if r.get("coherence") is not None]
                if vals:
                    avg_metrics["avg_coherence"] = sum(vals) / len(vals)
                    avg_metrics["sum_coherence"] = sum(vals)

        elif task_mode == "code" and eval_results:
            if "codebleu" in eval_results[0] and eval_results[0]["codebleu"] is not None:
                vals = []
                for r in eval_results:
                    v = r.get("codebleu")
                    if v is None:
                        continue
                    if isinstance(v, dict):
                        v = v.get("codebleu")
                    if isinstance(v, (int, float)):
                        vals.append(v)
                if vals:
                    avg_metrics["avg_codebleu"] = sum(vals) / len(vals)
                    avg_metrics["sum_codebleu"] = sum(vals)
            if "codebert" in eval_results[0] and eval_results[0]["codebert"] is not None:
                for key in ("f1", "precision", "recall"):
                    vals = [r["codebert"][key] for r in eval_results
                            if isinstance(r.get("codebert"), dict) and r["codebert"].get(key) is not None]
                    if vals:
                        avg_metrics[f"avg_codebert_{key}"] = sum(vals) / len(vals)
                        avg_metrics[f"sum_codebert_{key}"] = sum(vals)
        return avg_metrics


    def local_inference(self):
        batch = self._dequeue_batch()
        if not batch:
            return None

        self.is_idle = False
        start_time = time.time()

        instructions, inputs_text, outputs_text, prompts = self._build_prompts(batch)
        inputs = self._tokenize_prompts(prompts)
        input_tokens = self._count_inference_input_tokens(inputs)
        generated_texts, output_tokens = self._generate_texts(self.model, inputs, len(batch))
        loss, perplexity = self._compute_eval_loss(self.model, instructions, inputs_text, outputs_text)

        finished_time = time.time()
        process_time = finished_time - start_time

        s_batch = sum(
            1 for req in batch
            if finished_time - req.get("timestamp", float("inf")) <= req.get("ddl", float("inf"))
        )
        arrival_time, success_tokens = self._compute_slo_stats(batch, s_batch, output_tokens, start_time, finished_time)

        eval_results = self.select_evaluator(
            generated_texts, [r["output"] for r in batch], self.task_mode,
            instructions=[r["instruction"] for r in batch],
        )

        metrics_record = self._build_metrics_record(
            batch, s_batch, process_time, loss, perplexity,
            arrival_time, finished_time, input_tokens, output_tokens, success_tokens,
        )
        if eval_results:
            metrics_record.update(eval_results)
        self.serve_metrics.append(metrics_record)
        self.is_idle = True


    def _warmup_inference(self):
        warmup_request = {
            "request_id": f"warmup_{self.replica_id}",
            "instruction": "Introduce yourself.",
            "input": "", "output": "",
            "timestamp": time.time(), "ddl": 10.0,
        }
        with self.queue_lock:
            self.request_queue.append(warmup_request)
        self.local_inference()
        if self.serve_metrics:
            self.serve_metrics.pop()
        print(f"[Replica {self.replica_id}] warmup completed")


    def check_state(self):
        return self.state_manager.get_state(self.replica_id)

    def set_state(self, new_state):
        result = self.state_manager.update_state(self.replica_id, new_state)
        print(f"[Replica {self.replica_id}] {result}")
        return result

    def add_request(self, request):
        with self.queue_lock:
            self.request_queue.append(request)
        return True

    def get_queue_length(self):
        return len(self.request_queue)

    def get_metrics(self):
        return {
            "serve_metrics": self.serve_metrics,
            "training_metrics": self.training_metrics,
            "training_step_metrics": self.training_step_metrics,
        }

    def set_train_batch_size(self, v):
        self.train_batch_size = v

    def set_infer_batch_size(self, v):
        self.infer_batch_size = v

    def get_state_dict(self):
        return self.model.get_adapter_state_dict()

    def update_model(self, new_adapter_state_dict):
        self.model.load_state_dict(new_adapter_state_dict, strict=False)
        return f"[Replica {self.replica_id}] model updated"

    def download_model(self, global_model_state):
        if global_model_state is not None:
            self.model.load_state_dict(global_model_state, strict=False)
            print(f"[Replica {self.replica_id}] global model downloaded")

    def get_requests_for_migration(self, count):
        with self.queue_lock:
            count = min(count, len(self.request_queue))
            if count == 0:
                return []
            queue_list = list(self.request_queue)
            requests = queue_list[-count:]
            self.request_queue = deque(queue_list[:-count])
            return requests


    def _update_history_stats(self, slo_rate, queue_wait_avg):
        self._slo_recent.append(slo_rate)
        self._wait_recent.append(queue_wait_avg)
        self._slo_hist_sum += slo_rate
        self._slo_hist_cnt += 1
        self._wait_hist_sum += queue_wait_avg
        self._wait_hist_cnt += 1


    def local_train(self, max_steps=None, batch_size=None):
        if batch_size is not None:
            self.set_train_batch_size(batch_size)
        print(f"[Replica {self.replica_id}] local training started; batch_size={self.train_batch_size}, max_steps={max_steps}")

        train_start_time = time.time()
        train_args = SimpleTrainingArguments(
            batch_size=self.train_batch_size,
            max_steps=max_steps if max_steps else 100,
            gradient_accumulation_steps=8, logging_steps=10,
            save_steps=max_steps if max_steps else 100,
            learning_rate=5e-5, fp16=True,
            output_dir=f"./output/replica_{self.replica_id}",
        )
        trainer = SimpleTrainer(model=self.model, args=train_args,
                                train_dataset=self.train_dataset, tokenizer=self.tokenizer)

        dataloader_iter = trainer.get_train_dataloader_iter()
        update_steps = 0
        losses = []
        total_input_tokens = 0
        total_target_tokens = 0
        total_train_samples = 0
        while update_steps < max_steps:
            loss = trainer.train_one_update_step(dataloader_iter)
            if loss is None:
                dataloader_iter = trainer.get_train_dataloader_iter()
                continue
            losses.append(loss)
            update_steps += 1
            step_metrics = dict(getattr(trainer, "last_update_metrics", {}) or {})
            if step_metrics:
                step_metrics.update({
                    "replica_id": self.replica_id,
                    "update_step": update_steps,
                    "train_batch_size": self.train_batch_size,
                    "timestamp": step_metrics.get("update_end_time", time.time()),
                })
                self.training_step_metrics.append(step_metrics)
                total_input_tokens += step_metrics.get("update_input_tokens", 0)
                total_target_tokens += step_metrics.get("update_target_tokens", 0)
                total_train_samples += step_metrics.get("update_samples", 0)

        train_end_time = time.time()
        total_steps = max_steps * train_args.gradient_accumulation_steps
        wall_time = train_end_time - train_start_time

        initial_loss = losses[0]
        final_loss = losses[-1]
        train_loss = sum(losses) / len(losses)
        avg_loss_decrease = (initial_loss - final_loss) / total_steps if total_steps > 0 else 0

        gradient_noise = self._calculate_gradient_noise(trainer, self.train_batch_size)

        if initial_loss > 0:
            self.performance_score *= (1 + (initial_loss - final_loss) / initial_loss)
            print(f"[Replica {self.replica_id}] performance_score={self.performance_score:.4f}")

        training_metrics = {
            "replica_id": self.replica_id,
            "train_batch_size": self.train_batch_size,
            "max_steps": total_steps,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "train_loss": train_loss, "performance_score": self.performance_score,
            "avg_loss_decrease": avg_loss_decrease,
            "avg_iteration_time": wall_time / total_steps,
            "gradient_noise": gradient_noise,
            "wall_time_sec": wall_time,
            "steps_per_sec": total_steps / wall_time if wall_time > 0 else 0,
            "train_total_input_tokens": total_input_tokens,
            "train_total_target_tokens": total_target_tokens,
            "train_total_samples": total_train_samples,
            "train_input_tokens_per_sec": total_input_tokens / wall_time if wall_time > 0 else 0,
            "train_target_tokens_per_sec": total_target_tokens / wall_time if wall_time > 0 else 0,
            "train_samples_per_sec": total_train_samples / wall_time if wall_time > 0 else 0,
            "train_start_time": train_start_time, "train_end_time": train_end_time,
            "timestamp": time.time(),
        }
        self.training_metrics.append(training_metrics)
        print(f"[Replica {self.replica_id}] local training completed; wall_time={wall_time:.2f}s")
        return training_metrics

    def _calculate_gradient_noise(self, trainer, batch_size):
        try:
            model = trainer.model
            original_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

            small_bs = max(1, batch_size // 2)
            small_grads = self._compute_gradient(trainer, small_bs)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in original_params:
                        p.copy_(original_params[n])

            large_grads = self._compute_gradient(trainer, batch_size)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in original_params:
                        p.copy_(original_params[n])

            diff_norm = sum(torch.norm(small_grads[n] - large_grads[n]).item() ** 2
                           for n in small_grads if n in large_grads)
            grad_norm = sum(torch.norm(large_grads[n]).item() ** 2 for n in large_grads)

            if grad_norm > 0:
                noise_ratio = diff_norm / grad_norm
                return noise_ratio * (batch_size / (batch_size - small_bs))
            return 0.0
        except Exception as e:
            print(f"[Replica {self.replica_id}] gradient-noise estimation failed: {e}")
            return 0.1

    def _compute_gradient(self, trainer, batch_size):
        from torch.utils.data import DataLoader
        trainer.optimizer.zero_grad()
        dl = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        batch_data = next(iter(dl))
        batch = trainer._prepare_inputs(batch_data)
        outputs = trainer.model(**batch)
        outputs.loss.backward()
        return {n: p.grad.clone() for n, p in trainer.model.named_parameters()
                if p.requires_grad and p.grad is not None}



class DualAdapterReplica(Replica):
    def __init__(self, replica_id, model, train_dataset, tokenizer, args,
                 active_adapter_name, shadow_adapter_name,
                 state_manager=None, task_mode=None, quantization=None, baseline="CLIF"):
        self._weight_lock = threading.RLock()
        self.adapter_swap_count = 0
        self.active_adapter_name = active_adapter_name
        self.shadow_adapter_name = shadow_adapter_name
        self.baseline = baseline
        super().__init__(replica_id, model, train_dataset, tokenizer, args,
                         state_manager, task_mode, quantization)
        print(f"[DualAdapterReplica {self.replica_id}] buffer: {self.model.device}")

    def _adapter_alias(self):
        return "__clif_global_adapter__"

    def _normalize_adapter_state_dict(self, state_dict, adapter_name):
        alias = self._adapter_alias()
        return {k.replace(adapter_name, alias): v for k, v in state_dict.items()}

    def _denormalize_adapter_state_dict(self, state_dict, adapter_name):
        alias = self._adapter_alias()
        return {k.replace(alias, adapter_name): v for k, v in state_dict.items()}

    def _get_named_adapter_state_dict(self, adapter_name):
        try:
            state = self.model.get_adapter_state_dict(adapter_name=adapter_name)
        except TypeError:
            previous_adapter = getattr(self.model, "active_adapter", None)
            try:
                self.model.set_adapter(adapter_name)
                state = self.model.get_adapter_state_dict()
            finally:
                if previous_adapter is not None:
                    try:
                        self.model.set_adapter(previous_adapter)
                    except Exception:
                        pass
        return {k: v.detach().clone() for k, v in state.items()}

    def get_state_dict(self, adapter_name=None):
        adapter_name = adapter_name or self.active_adapter_name
        with self._weight_lock:
            state = self._get_named_adapter_state_dict(adapter_name)
        print(f"[DualAdapterReplica {self.replica_id}] export adapter: {adapter_name}")
        return self._normalize_adapter_state_dict(state, adapter_name)

    def update_model(self, new_adapter_state_dict, adapter_name=None):
        adapter_name = adapter_name or self.active_adapter_name
        with self._weight_lock:
            mapped_state = self._denormalize_adapter_state_dict(new_adapter_state_dict, adapter_name)
            self.model.load_state_dict(mapped_state, strict=False)
        return f"[DualAdapterReplica {self.replica_id}] adapter {adapter_name} updated"

    def local_inference(self):
        batch = self._dequeue_batch()
        if not batch:
            return None

        self.is_idle = False
        start_time = time.time()

        from utils.thread_adapter import adapter_context

        with adapter_context(self.active_adapter_name):
            instructions, inputs_text, outputs_text, prompts = self._build_prompts(batch)
            inputs = self._tokenize_prompts(prompts)
            input_tokens = self._count_inference_input_tokens(inputs)
            generated_texts, output_tokens = self._generate_texts(self.model, inputs, len(batch))
            loss, perplexity = self._compute_eval_loss(
                self.model, instructions, inputs_text, outputs_text,
            )

        finished_time = time.time()
        process_time = finished_time - start_time

        s_batch = sum(
            1 for req in batch
            if finished_time - req.get("timestamp", float("inf")) <= req.get("ddl", float("inf"))
        )
        arrival_time, success_tokens = self._compute_slo_stats(batch, s_batch, output_tokens, start_time, finished_time)

        batch_size_actual = len(batch)
        slo_rate = s_batch / batch_size_actual if batch_size_actual > 0 else 0.0
        queue_waits = [start_time - req.get("timestamp", start_time) for req in batch]
        queue_wait_avg = sum(queue_waits) / batch_size_actual if batch_size_actual > 0 else 0.0
        self._update_history_stats(slo_rate, queue_wait_avg)

        eval_results = self.select_evaluator(
            generated_texts, [r["output"] for r in batch], self.task_mode,
            instructions=[r["instruction"] for r in batch],
        )

        metrics_record = self._build_metrics_record(
            batch, s_batch, process_time, loss, perplexity,
            arrival_time, finished_time, input_tokens, output_tokens, success_tokens,
        )
        if eval_results:
            metrics_record.update(eval_results)
        self.serve_metrics.append(metrics_record)
        self.is_idle = True

    def local_train(self, max_steps=None, gradient_accumulation_steps=8, batch_size=None,
                    swap_strategy="fixed", swap_interval=50, swap_loss_delta=0.05):
        current_state = self.check_state()
        if current_state not in [ReplicaState.TRAINING, ReplicaState.COMBINED]:
            print(f"[DualAdapterReplica {self.replica_id}] skip training in state={current_state}")
            return None

        if batch_size is not None:
            self.set_train_batch_size(batch_size)
        print(f"[DualAdapterReplica {self.replica_id}] local training started; "
              f"batch_size={self.train_batch_size}, max_steps={max_steps}, "
              f"swap: {swap_strategy}({'interval=' + str(swap_interval) if swap_strategy == 'fixed' else 'delta=' + str(swap_loss_delta)})")

        from utils.thread_adapter import adapter_context

        train_start_time = time.time()
        train_args = SimpleTrainingArguments(
            batch_size=self.train_batch_size,
            max_steps=max_steps if max_steps else 100,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=10, save_steps=max_steps if max_steps else 100,
            learning_rate=5e-5, fp16=True,
            output_dir=f"./output/replica_{self.replica_id}",
        )
        trainer = SimpleTrainer(model=self.model, args=train_args,
                                train_dataset=self.train_dataset, tokenizer=self.tokenizer)
        dataloader_iter = trainer.get_train_dataloader_iter()
        update_steps = 0
        losses = []
        adapter_update_time = 0
        adapter_swap_count = 0
        latest_training_promoted = True
        total_input_tokens = 0
        total_target_tokens = 0
        total_train_samples = 0
        last_swap_loss = None
        if self.baseline == "CLIF":
            adapter_update_time += self.update_shadow_adapter()

        while update_steps < max_steps:
            should_swap = False
            if self.baseline == "CLIF":
                if swap_strategy == "fixed":
                    should_swap = (update_steps % max(1, swap_interval) == 0)
                elif swap_strategy == "smart":
                    if last_swap_loss is None:
                        should_swap = True
                    elif losses:
                        current_loss = losses[-1]
                        drop_ratio = (last_swap_loss - current_loss) / (last_swap_loss + 1e-8)
                        should_swap = (drop_ratio >= swap_loss_delta)

            should_swap = False

            with adapter_context(self.shadow_adapter_name):
                loss = trainer.train_one_update_step(dataloader_iter)

            if loss is None:
                dataloader_iter = trainer.get_train_dataloader_iter()
                continue
            losses.append(loss)
            update_steps += 1
            latest_training_promoted = False
            step_metrics = dict(getattr(trainer, "last_update_metrics", {}) or {})
            if step_metrics:
                step_metrics.update({
                    "replica_id": self.replica_id,
                    "update_step": update_steps,
                    "train_batch_size": self.train_batch_size,
                    "timestamp": step_metrics.get("update_end_time", time.time()),
                    "active_adapter_name": self.active_adapter_name,
                    "shadow_adapter_name": self.shadow_adapter_name,
                    "swap_strategy": swap_strategy,
                })
                self.training_step_metrics.append(step_metrics)
                total_input_tokens += step_metrics.get("update_input_tokens", 0)
                total_target_tokens += step_metrics.get("update_target_tokens", 0)
                total_train_samples += step_metrics.get("update_samples", 0)

            should_promote = False
            if self.baseline == "CLIF":
                if swap_strategy == "fixed":
                    should_promote = (update_steps % max(1, swap_interval) == 0)
                elif swap_strategy == "smart":
                    if last_swap_loss is None:
                        should_promote = True
                    else:
                        drop_ratio = (last_swap_loss - loss) / (last_swap_loss + 1e-8)
                        should_promote = (drop_ratio >= swap_loss_delta)

            if should_promote and self.baseline == "CLIF":
                self.exchange_adapter()
                latest_training_promoted = True
                adapter_swap_count += 1
                if swap_strategy == "smart":
                    last_swap_loss = loss
                if update_steps < max_steps:
                    adapter_update_time += self.update_shadow_adapter()

        final_promote_time = 0.0
        if self.baseline == "CLIF" and not latest_training_promoted:
            promote_start = time.time()
            self.exchange_adapter()
            final_promote_time = time.time() - promote_start
            adapter_swap_count += 1
            print(
                f"[DualAdapterReplica {self.replica_id}] final promote shadow -> active, "
                f"active_adapter={self.active_adapter_name}"
            )

        train_end_time = time.time()
        total_steps = max_steps * train_args.gradient_accumulation_steps
        wall_time = train_end_time - train_start_time

        initial_loss = losses[0]
        final_loss = losses[-1]
        train_loss = sum(losses) / len(losses)
        avg_loss_decrease = (initial_loss - final_loss) / total_steps if total_steps > 0 else 0
        gradient_noise = self._calculate_gradient_noise(trainer, self.train_batch_size)

        if initial_loss > 0:
            self.performance_score *= (1 + (initial_loss - final_loss) / initial_loss)

        training_metrics = {
            "replica_id": self.replica_id,
            "train_batch_size": self.train_batch_size,
            "max_steps": total_steps,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "train_loss": train_loss, "performance_score": self.performance_score,
            "avg_loss_decrease": avg_loss_decrease,
            "avg_iteration_time": wall_time / total_steps,
            "gradient_noise": gradient_noise,
            "wall_time_sec": wall_time,
            "steps_per_sec": total_steps / wall_time if wall_time > 0 else 0,
            "train_total_input_tokens": total_input_tokens,
            "train_total_target_tokens": total_target_tokens,
            "train_total_samples": total_train_samples,
            "train_input_tokens_per_sec": total_input_tokens / wall_time if wall_time > 0 else 0,
            "train_target_tokens_per_sec": total_target_tokens / wall_time if wall_time > 0 else 0,
            "train_samples_per_sec": total_train_samples / wall_time if wall_time > 0 else 0,
            "train_start_time": train_start_time, "train_end_time": train_end_time,
            "timestamp": time.time(),
            "adapter_swap_count": adapter_swap_count,
            "avg_adapter_update_time": adapter_update_time / max(adapter_swap_count, 1),
            "total_adapter_update_time": adapter_update_time,
            "final_promote_time": final_promote_time,
            "export_adapter_name": self.active_adapter_name,
            "swap_strategy": swap_strategy,
        }
        self.training_metrics.append(training_metrics)
        print(f"[DualAdapterReplica {self.replica_id}] local training completed; "
              f"wall_time={wall_time:.2f}s, adapter_swaps={adapter_swap_count}")
        return training_metrics

    def update_shadow_adapter(self):
        start_time = time.time()
        with self._weight_lock:
            update_state_dict = {}
            state_dict = self.model.state_dict()
            for name, param in state_dict.items():
                if self.active_adapter_name in name:
                    new_name = name.replace(self.active_adapter_name, self.shadow_adapter_name)
                    update_state_dict[new_name] = param.clone()
            self.model.load_state_dict(update_state_dict, strict=False)
        return time.time() - start_time

    def exchange_adapter(self):
        self.active_adapter_name, self.shadow_adapter_name = self.shadow_adapter_name, self.active_adapter_name

    def download_model(self, global_model_state):
        with self._weight_lock:
            self.update_model(global_model_state, adapter_name=self.shadow_adapter_name)
            self.exchange_adapter()
            print(
                f"[DualAdapterReplica {self.replica_id}] global adapter downloaded, "
                f"active_adapter={self.active_adapter_name}"
            )
