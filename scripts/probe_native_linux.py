"""Print the native Linux/CUDA runtime used by a DeltaServe job."""

from __future__ import annotations

import importlib.util
import json
import platform
import socket
import sys


def main() -> None:
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "vllm_installed": importlib.util.find_spec("vllm") is not None,
        "cuda_bindings_installed": importlib.util.find_spec("cuda.bindings") is not None,
    }
    import torch

    result.update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    if torch.cuda.is_available():
        result.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    try:
        import vllm

        result["vllm"] = vllm.__version__
    except Exception as exc:  # pragma: no cover - only used for diagnostics
        result["vllm_import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
