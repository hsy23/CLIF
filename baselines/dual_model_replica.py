
import time
import torch
import threading
from common.trainer import SimpleTrainer, SimpleTrainingArguments
from core.replica import Replica


class DualModelReplica(Replica):

    def __init__(self, replica_id, model_infer, model_train, train_dataset,
                 tokenizer, args, adapter_name_infer, adapter_name_train,
                 state_manager=None, task_mode=None, quantization=None):
        self.model_lock = threading.Lock()
        self.model_swap_count = 0

        self.model_infer = model_infer
        self.model_train = model_train
        self.adapter_name_infer = adapter_name_infer
        self.adapter_name_train = adapter_name_train

        super().__init__(replica_id, model_infer, train_dataset, tokenizer, args,
                         state_manager, task_mode, quantization)
        print(f"[DualModelReplica {self.replica_id}] initialized; "
              f"inference_device={self.model_infer.device}, training_device={self.model_train.device}")


    def local_inference(self):
        batch = self._dequeue_batch()
        if not batch:
            return None

        self.is_idle = False
        start_time = time.time()

        instructions, inputs_text, outputs_text, prompts = self._build_prompts(batch)
        inputs = self._tokenize_prompts(prompts, device=self.model_infer.device)
        generated_texts, output_tokens = self._generate_texts(self.model_infer, inputs, len(batch))
        loss, perplexity = self._compute_eval_loss(
            self.model_infer, instructions, inputs_text, outputs_text,
            device=self.model_infer.device,
        )

        finished_time = time.time()
        process_time = finished_time - start_time

        s_batch = sum(
            1 for req in batch
            if finished_time - req.get("timestamp", float("inf")) <= req.get("ddl", float("inf"))
        )
        arrival_time, success_tokens = self._compute_slo_stats(
            batch, s_batch, output_tokens, start_time, finished_time,
        )

        eval_results = self.select_evaluator(
            generated_texts, [r["output"] for r in batch], self.task_mode,
            instructions=[r["instruction"] for r in batch],
        )

        metrics_record = self._build_metrics_record(
            batch, s_batch, process_time, loss, perplexity,
            arrival_time, finished_time, output_tokens, success_tokens,
        )
        if eval_results:
            metrics_record.update(eval_results)
        self.serve_metrics.append(metrics_record)
        self.is_idle = True


    def local_train(self, max_steps=None, gradient_accumulation_steps=8, batch_size=None):
        if batch_size is not None:
            self.set_train_batch_size(batch_size)
        print(f"[DualModelReplica {self.replica_id}] local training started; "
              f"batch_size={self.train_batch_size}, max_steps={max_steps}")

        with self.model_lock:
            self.model_train.load_state_dict(self.model_infer.state_dict(), strict=False)

        train_start_time = time.time()
        train_args = SimpleTrainingArguments(
            batch_size=self.train_batch_size,
            max_steps=max_steps if max_steps else 100,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=10, save_steps=max_steps if max_steps else 100,
            learning_rate=5e-5, fp16=True,
            output_dir=f"./output/replica_{self.replica_id}",
        )
        trainer = SimpleTrainer(model=self.model_train, args=train_args,
                                train_dataset=self.train_dataset, tokenizer=self.tokenizer)
        dataloader_iter = trainer.get_train_dataloader_iter()
        update_steps = 0
        losses = []
        model_update_time = 0
        model_swap_count = 0

        while update_steps < max_steps:
            loss = trainer.train_one_update_step(dataloader_iter)
            if loss is None:
                dataloader_iter = trainer.get_train_dataloader_iter()
                continue
            losses.append(loss)
            update_steps += 1

        model_update_time += self.exchange_model_and_adapter()
        model_swap_count += 1

        train_end_time = time.time()
        total_steps = max_steps * train_args.gradient_accumulation_steps
        wall_time = train_end_time - train_start_time

        initial_loss = losses[0]
        final_loss = losses[-1]
        train_loss = sum(losses) / len(losses)
        avg_loss_decrease = (initial_loss - final_loss) / total_steps if total_steps > 0 else 0

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
            "wall_time_sec": wall_time,
            "steps_per_sec": total_steps / wall_time if wall_time > 0 else 0,
            "train_start_time": train_start_time, "train_end_time": train_end_time,
            "timestamp": time.time(),
            "model_swap_count": model_swap_count,
            "avg_model_update_time": model_update_time / max(model_swap_count, 1),
            "total_model_update_time": model_update_time,
        }
        self.training_metrics.append(training_metrics)
        print(f"[DualModelReplica {self.replica_id}] local training completed; wall_time={wall_time:.2f}s")
        return training_metrics


    def exchange_model_and_adapter(self):
        start_time = time.time()
        with self.model_lock:
            self.model_infer, self.model_train = self.model_train, self.model_infer
            self.adapter_name_infer, self.adapter_name_train = self.adapter_name_train, self.adapter_name_infer
            self.model = self.model_infer
        return time.time() - start_time


    def get_state_dict(self):
        return self.model_infer.get_adapter_state_dict()

    def update_model(self, new_adapter_state_dict):
        self.model_infer.load_state_dict(new_adapter_state_dict, strict=False)
        return f"[DualModelReplica {self.replica_id}] model updated"

    def download_model(self, global_model_state):
        if global_model_state is not None:
            with self.model_lock:
                self.model_infer.load_state_dict(global_model_state, strict=False)
            print(f"[DualModelReplica {self.replica_id}] global model downloaded")
            return True
        return False
