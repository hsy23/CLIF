import argparse
import logging
import sys
import warnings

from main import main

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils.checkpoint.*")
warnings.filterwarnings("ignore", message="There is no reference data-flows extracted from the whole corpus")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIF: a continuous learning and inference framework for PEFT serving.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--baseline", type=str, default="CLIF", choices=["CLIF"], help="System mode.")
    parser.add_argument("--dispatcher", type=str, default=None, choices=["subflow"], help="Dispatcher policy.")
    parser.add_argument("--enable_dual_adapter", action="store_true", help="Enable dual-adapter replicas.")
    parser.add_argument("--force_quantization", type=str, choices=["4bit", "8bit", "none"], default="none", help="Quantization mode.")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model name or local path.")
    parser.add_argument(
        "--train_datasets_conv",
        type=str,
        nargs="+",
        default=["tatsu-lab/alpaca", "crumb/Clean-Instruct-3M", "taskydata/GPTeacher-General-Instruct", "hakurei/open-instruct-v1"],
        help="Instruction-tuning datasets for conversational tasks.",
    )
    parser.add_argument("--train_datasets_code", type=str, nargs="+", default=["iamtarun/code_instructions_120k_alpaca"], help="Instruction-tuning datasets for code tasks.")
    parser.add_argument("--max_train_samples", type=int, default=10000, help="Maximum local training samples per replica.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory.")
    parser.add_argument("--output_subdir", type=str, default=None, help="Optional output subdirectory.")
    parser.add_argument("--num_replicas", type=int, default=8, help="Number of replicas.")
    parser.add_argument("--replica_gpus", type=str, default=None, help='Replica-to-GPU mapping as JSON, for example "[[0,1],[2,3]]".')

    parser.add_argument("--train_size", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--infer_size", type=float, default=0.2, help="Inference split ratio.")
    parser.add_argument("--train_batch_size", type=int, nargs="+", default=[8] * 8, help="Initial training batch size per replica.")
    parser.add_argument("--infer_batch_size", type=int, nargs="+", default=[6] * 8, help="Initial inference batch size per replica.")
    parser.add_argument("--infer_length", type=int, default=128, help="Maximum input prompt length.")
    parser.add_argument("--generation_strategy", type=str, default="greedy", choices=["greedy", "sampling", "beam_search"], help="Generation strategy.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum generated tokens per request.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam count for beam search.")
    parser.add_argument("--low_rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--initial_score", type=float, nargs="+", default=[1.0] * 8, help="Initial quality score per replica.")

    parser.add_argument("--fl_rounds", type=int, default=5, help="Number of federated fine-tuning rounds.")
    parser.add_argument("--local_steps", type=int, default=100, help="Local optimizer steps per round.")
    parser.add_argument("--adapter_swap_strategy", type=str, default="fixed", choices=["fixed", "smart"], help="Dual-adapter promotion policy.")
    parser.add_argument("--adapter_swap_interval", type=int, default=50, help="Promotion interval for fixed dual-adapter swaps.")
    parser.add_argument("--adapter_swap_loss_delta", type=float, default=0.05, help="Loss-improvement threshold for smart swaps.")
    parser.add_argument("--min_train_batch", type=int, default=4, help="Minimum training batch size.")
    parser.add_argument("--max_train_batch", type=int, default=8, help="Maximum training batch size.")
    parser.add_argument("--min_infer_batch", type=int, default=4, help="Minimum inference batch size.")
    parser.add_argument("--max_infer_batch", type=int, default=8, help="Maximum inference batch size.")

    parser.add_argument("--request_interval", type=float, default=0.5, help="Mean interval for fixed request generation.")
    parser.add_argument("--concurrent_requests", type=int, default=20, help="Concurrent requests for fixed generation.")
    parser.add_argument("--request_pattern", type=str, default="trace", choices=["trace", "fixed"], help="Request-generation mode.")
    parser.add_argument("--ddl", type=float, default=1.4, help="End-to-end request SLO in seconds.")
    parser.add_argument("--max_requests", type=int, default=80000, help="Maximum generated requests.")
    parser.add_argument("--request_start_date", type=str, default="2024-05-12 10:00:00", help="Trace replay start timestamp.")
    parser.add_argument("--scale_up", type=float, default=1.0, help="Trace replay speedup factor.")
    parser.add_argument("--task_mode", type=str, default="code", choices=["conv", "code"], help="Task family.")
    parser.add_argument("--request_trace_conv", type=str, default="./data/AzureLLMInferenceTrace_conv_1week.csv", help="Conversation trace CSV.")
    parser.add_argument("--request_trace_code", type=str, default="./data/AzureLLMInferenceTrace_code_1week.csv", help="Code trace CSV.")

    parser.add_argument("--idle_strategy", type=str, default="ewma", choices=["slo", "ewma", "fixed_idle"], help="Policy for identifying lightly loaded replicas.")
    parser.add_argument("--test_light_count", type=int, default=2, help="Number of fixed IDLE replicas when using fixed_idle.")
    parser.add_argument("--ewma_alpha", type=float, default=0.3, help="EWMA smoothing factor.")
    parser.add_argument("--idle_timeout", type=int, default=5, help="Seconds before returning unused IDLE replicas to SERVING.")
    parser.add_argument("--run_time", type=int, default=600, help="Run duration in seconds.")
    parser.add_argument("--fl_start_time", type=int, default=60, help="Delay before launching federated fine-tuning.")
    parser.add_argument("--fl_min_participants", type=int, default=2, help="Minimum participants required for a federated round.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 50)
    print("CLIF: Continuous Learning and Inference Framework")
    print("=" * 50)

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as exc:
        print(f"\nRun failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
