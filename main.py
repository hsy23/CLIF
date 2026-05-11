
import os
import time
import threading
import torch
import pandas as pd
from datetime import datetime

from core.replica import Replica, DualAdapterReplica
from core.dispatcher import SubflowDispatcher
from core.coordinator import Coordinator
from core.fl_launcher import FederatedLearningLauncher
from baselines.dispatchers import RoundDispatcher, dLoRADispatcher, ShepherdDispatcher
from baselines.dual_model_replica import DualModelReplica
from common.state import StateManagement, ReplicaState
from common.monitor import GPUMonitor
from common.request_generator import RequestGenerator
from common.model_loader import (
    load_tokenizer, load_multi_token_datasets,
    load_model_instance, load_model_instance_multi_GPU,
    load_dual_model_instance, parse_replica_gpus,
)

DISPATCHER_MAP = {
    "subflow": lambda reps, ddl: SubflowDispatcher(reps, ddl=ddl),
    "round": lambda reps, ddl: RoundDispatcher(reps, ddl=ddl),
    "dLoRA": lambda reps, ddl: dLoRADispatcher(reps, ddl=ddl),
    "Shepherd": lambda reps, ddl: ShepherdDispatcher(reps, ddl=ddl),
}

BASELINE_DEFAULT_DISPATCHER = {
    "CLIF": "subflow",
    "dLoRA": "dLoRA",
    "Shepherd": "Shepherd",
    "DualModel": "round",
    "test": "round",
}

FL_BASELINES = {"CLIF", "DualModel", "test"}


def main(args):
    if args.output_subdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"run_{ts}")
    else:
        output_dir = os.path.join(args.output_dir, args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        gpu_count = torch.cuda.device_count()
        f.write(f"GPU: {gpu_count}\n")
        for i in range(gpu_count):
            f.write(f"GPU {i} : {torch.cuda.get_device_name(i)}\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print(f"[Main] Starting {args.baseline}; output_dir={output_dir}")

    gpu_monitor = GPUMonitor(result_dir=output_dir, gpu_ids=range(args.num_replicas),
                             interval=0.5, p_interval=50)
    gpu_monitor.start()

    tokenizer = load_tokenizer(args.model_name)
    ds_list = args.train_datasets_conv if args.task_mode == "conv" else args.train_datasets_code
    train_datasets, inference_dataset = load_multi_token_datasets(
        ds_list, args.train_size, args.infer_size, args.num_replicas,
        tokenizer, max_train_samples=args.max_train_samples)

    idle_strategy = getattr(args, "idle_strategy", "slo")
    state_manager = StateManagement(
        idle_strategy=idle_strategy,
        idle_timeout=getattr(args, "idle_timeout", 5),
        ewma_alpha=getattr(args, "ewma_alpha", 0.3),
    )
    state_manager.reset_state_history()

    quantization = None

    use_dual_adapter = args.enable_dual_adapter
    if use_dual_adapter and args.baseline == "CLIF":
        print("[Main] Enabling dual-adapter mode")
        from utils.thread_adapter import install_thread_local_adapter_patch
        install_thread_local_adapter_patch()
    elif args.baseline == "CLIF":
        print("[Main] Using standard single-adapter replicas")

    replica_gpu_assignments = parse_replica_gpus(args)
    replicas = []
    for i in range(args.num_replicas):
        gpu_list = replica_gpu_assignments[i]
        device = torch.device(f"cuda:{gpu_list[0]}")
        print(f"[Main] Replica {i} GPU: {gpu_list}")

        if args.baseline == "CLIF":
            if use_dual_adapter:
                active = f"active_adapter_{i}"
                shadow = f"shadow_adapter_{i}"
                model = load_dual_model_instance(
                    args.model_name, args.low_rank, device,
                    active, shadow, i, quantization, args.force_quantization)
                replica = DualAdapterReplica(
                    replica_id=i, model=model, train_dataset=train_datasets[i],
                    tokenizer=tokenizer, args=args,
                    active_adapter_name=active, shadow_adapter_name=shadow,
                    state_manager=state_manager, task_mode=args.task_mode,
                    baseline="CLIF")
            else:
                model = (_load_model(args, gpu_list, device, quantization))
                replica = Replica(
                    replica_id=i, model=model, train_dataset=train_datasets[i],
                    tokenizer=tokenizer, args=args,
                    state_manager=state_manager, task_mode=args.task_mode)

        elif args.baseline in ("dLoRA", "Shepherd"):
            model = _load_model(args, gpu_list, device, quantization)
            replica = Replica(
                replica_id=i, model=model, train_dataset=train_datasets[i],
                tokenizer=tokenizer, args=args,
                state_manager=state_manager, task_mode=args.task_mode)

        elif args.baseline == "DualModel":
            model_infer = _load_model(args, gpu_list, device, quantization)
            model_train = _load_model(args, gpu_list, device, quantization)
            replica = DualModelReplica(
                replica_id=i, model_infer=model_infer, model_train=model_train,
                train_dataset=train_datasets[i], tokenizer=tokenizer, args=args,
                adapter_name_infer=f"infer_adapter_{i}",
                adapter_name_train=f"train_adapter_{i}",
                state_manager=state_manager, task_mode=args.task_mode)

        elif args.baseline == "test":
            model = load_model_instance_multi_GPU(
                args.model_name, args.low_rank, device=device,
                device_map=gpu_list, force_quantization=args.force_quantization)
            replica = Replica(
                replica_id=i, model=model, train_dataset=train_datasets[i],
                tokenizer=tokenizer, args=args,
                state_manager=state_manager, task_mode=args.task_mode)

        replica._warmup_inference()
        replicas.append(replica)

    ddl_eff = args.ddl * 0.9
    dispatcher_name = getattr(args, "dispatcher", None) or BASELINE_DEFAULT_DISPATCHER[args.baseline]
    dispatcher = DISPATCHER_MAP[dispatcher_name](replicas, ddl_eff)
    print(f"[Main] Dispatcher policy: {dispatcher_name}")
    dispatcher.gpu_monitor = gpu_monitor

    trace = None
    if args.request_pattern == "trace":
        trace = args.request_trace_conv if args.task_mode == "conv" else args.request_trace_code
    request_generator = RequestGenerator(
        inference_dataset=inference_dataset, dispatcher=dispatcher,
        concurrent_requests=args.concurrent_requests,
        mean_interval=args.request_interval, ddl=args.ddl,
        max_requests=args.max_requests, request_trace=trace,
        start_date=args.request_start_date,
        duration_time=args.run_time + 30, scale_up=args.scale_up,
        pattern=args.request_pattern)

    if args.request_pattern == "trace":
        request_generator.load_trace_range()
    request_generator.start()
    time.sleep(0.01)
    dispatcher.start()

    for rep in replicas:
        t = threading.Thread(target=_inference_loop, args=(rep,), daemon=True)
        t.start()

    coordinator = None
    fl_launcher = None
    if args.baseline == "CLIF":
        coordinator = Coordinator(
            min_train_batch=args.min_train_batch, max_train_batch=args.max_train_batch,
            min_infer_batch=args.min_infer_batch, max_infer_batch=args.max_infer_batch,
            ddl=ddl_eff, result_dir=output_dir)
    if args.baseline in FL_BASELINES:
        fl_launcher = FederatedLearningLauncher(
            replicas=replicas, request_dispatcher=dispatcher,
            coordinator=coordinator, state_manager=state_manager,
            local_steps=args.local_steps, rounds=args.fl_rounds,
            baseline=args.baseline,
            swap_strategy=getattr(args, "adapter_swap_strategy", "fixed"),
            swap_interval=getattr(args, "adapter_swap_interval", 50),
            swap_loss_delta=getattr(args, "adapter_swap_loss_delta", 0.05),
            min_participants=getattr(args, "fl_min_participants", 1))
    else:
        print(f"[Main] {args.baseline} runs without federated fine-tuning")

    try:
        print(f"[Main] {datetime.now():%Y-%m-%d %H:%M:%S} {args.run_time}s")
        start = time.time()
        fl_started = False
        fl_trigger = start + args.fl_start_time

        while time.time() - start < args.run_time:
            if not fl_started and fl_launcher and time.time() >= fl_trigger:
                if idle_strategy == "fixed_idle":
                    n = min(getattr(args, "test_light_count", 2), len(replicas))
                    for rep in replicas[:n]:
                        rep.set_state(ReplicaState.IDLE)
                    print(f"[Main] fixed_idle marked {n} replicas as IDLE")
                elif args.baseline == "test":
                    replicas[0].set_state(ReplicaState.IDLE)
                elif args.baseline == "DualModel":
                    for rep in replicas[:min(args.train_replicas_num, len(replicas))]:
                        rep.set_state(ReplicaState.IDLE)

                print("[Main] Starting federated fine-tuning")
                fl_launcher.start_fl()
                fl_started = True

            elapsed = int(time.time() - start)
            if elapsed % 100 == 0:
                print(f"[Main] Running... elapsed={elapsed}s")
            time.sleep(1)

    except KeyboardInterrupt:
        print("[Main] Interrupted")

    finally:
        print("[Main] Stopping components")
        request_generator.stop()
        dispatcher.stop()
        gpu_monitor.stop()
        if fl_launcher:
            fl_launcher.stop_fl()
            save_metrics(args, output_dir, replicas, dispatcher, request_generator, fl_launcher)
        else:
            save_metrics(args, output_dir, replicas, dispatcher, request_generator)
        print(f"[Main] Finished {args.baseline}; output_dir={output_dir}")



def _load_model(args, gpu_list, device, quantization):
    if len(gpu_list) > 1:
        return load_model_instance_multi_GPU(
            args.model_name, args.low_rank, device=None,
            device_map=gpu_list, force_quantization=args.force_quantization)
    return load_model_instance(
        args.model_name, args.low_rank, device, quantization, args.force_quantization)


def _inference_loop(replica):
    while True:
        replica.local_inference()
        time.sleep(0.0001)


def save_metrics(args, output_dir, replicas, dispatcher, request_generator, fl_launcher=None):
    replica_metrics = [r.get_metrics() for r in replicas]

    serve_metrics = []
    for rm in replica_metrics:
        serve_metrics.extend(rm.get("serve_metrics", []))
    pd.DataFrame(serve_metrics).to_excel(os.path.join(output_dir, "serve_metrics.xlsx"), index=False)

    train_metrics = []
    for rm in replica_metrics:
        train_metrics.extend(rm.get("training_metrics", []))
    pd.DataFrame(train_metrics).to_excel(os.path.join(output_dir, "train_metrics.xlsx"), index=False)

    train_step_metrics = []
    for rm in replica_metrics:
        train_step_metrics.extend(rm.get("training_step_metrics", []))
    pd.DataFrame(train_step_metrics).to_excel(os.path.join(output_dir, "train_step_metrics.xlsx"), index=False)

    state_columns = [
        "timestamp", "relative_time_sec", "datetime", "replica_id",
        "old_state", "new_state", "event", "changed",
    ]
    state_history = []
    if replicas:
        state_manager = getattr(replicas[0], "state_manager", None)
        if state_manager is not None and hasattr(state_manager, "get_state_history"):
            state_history = state_manager.get_state_history()
    pd.DataFrame(state_history, columns=state_columns).to_excel(
        os.path.join(output_dir, "state_metrics.xlsx"), index=False)

    pd.DataFrame(dispatcher.get_metrics()).to_excel(
        os.path.join(output_dir, "dispatch_metrics.xlsx"), index=False)

    gen_metrics = request_generator.get_metrics()
    pd.DataFrame(gen_metrics.get("metrics")).to_excel(
        os.path.join(output_dir, "request_gen_metrics.xlsx"), index=False)

    fl_metrics = None
    if fl_launcher:
        fl_metrics = fl_launcher.get_metrics()
        pd.DataFrame(fl_metrics.get("round_metrics")).to_excel(
            os.path.join(output_dir, "fl_round_metrics.xlsx"), index=False)

    if not serve_metrics:
        print("[Metrics] No serving metrics were collected")
        return

    total_service_time = serve_metrics[-1]["finished_time"] - serve_metrics[0]["arrival_time"]
    serve_df = pd.DataFrame(serve_metrics)
    serve_df["eval_loss"] = serve_df["eval_loss"].ffill()

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "request_generation_count": gen_metrics.get("request_count", 0) if gen_metrics else 0,
        "total_requests": serve_df["infer_batch_size"].sum() if not serve_df.empty else 0,
        "total_success_requests": serve_df["s_batch"].sum() if not serve_df.empty else 0,
        "throughput": serve_df["infer_batch_size"].sum() / total_service_time if not serve_df.empty else 0,
        "gootput": serve_df["s_batch"].sum() / total_service_time if not serve_df.empty else 0,
        "avg_response_quality": (serve_df["s_batch"] / serve_df["eval_loss"]).sum() / serve_df["s_batch"].sum() if not serve_df.empty else 0,
        "token_throughput": serve_df["output_tokens"].sum() / total_service_time if not serve_df.empty else 0,
        "token_goodput": serve_df["success_tokens"].sum() / total_service_time if not serve_df.empty else 0,
        "total_fl_rounds": len(fl_metrics.get("round_metrics", [])) if fl_metrics else 0,
        "num_replicas": len(replicas),
    }

    if args.task_mode == "conv":
        if "avg_bleu" in serve_df.columns:
            summary["avg_bleu"] = serve_df["avg_bleu"].mean()
            summary["goodput_avg_bleu"] = (serve_df["s_batch"] * serve_df["avg_bleu"]).sum() * 100 / total_service_time
        if "sum_bleu" in serve_df.columns:
            summary["wavg_bleu"] = serve_df["sum_bleu"].sum() / serve_df["infer_batch_size"].sum()
    elif args.task_mode == "code":
        if "avg_codebleu" in serve_df.columns:
            summary["avg_codebleu"] = serve_df["avg_codebleu"].mean()
            summary["goodput_avg_codebleu"] = (serve_df["s_batch"] * serve_df["avg_codebleu"]).sum() * 100 / total_service_time
        if "sum_codebleu" in serve_df.columns:
            summary["wavg_codebleu"] = serve_df["sum_codebleu"].sum() / serve_df["infer_batch_size"].sum()

    pd.DataFrame([summary]).to_excel(os.path.join(output_dir, "summary.xlsx"), index=False)
    print(f"[Metrics] Saved metrics to {output_dir}")
