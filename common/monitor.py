import time
import threading
import torch
import pandas as pd
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


class GPUMonitor(threading.Thread):
    def __init__(self, result_dir, gpu_ids=[0], interval=1, p_interval=50):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.result_dir = result_dir
        self.interval = interval
        self.p_interval = p_interval

        self.data = []
        self.stop_event = threading.Event()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lock = threading.Lock()

        os.makedirs(result_dir, exist_ok=True)
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 8))
        self.line_util = None
        self.line_mem = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                for gpu_id in self.gpu_ids:
                    try:
                        result = subprocess.check_output(
                            ["nvidia-smi",
                             "--query-gpu=utilization.gpu,memory.used,memory.total",
                             "--format=csv,nounits,noheader",
                             "-i", str(gpu_id)],
                            encoding="utf-8",
                        )
                        gpu_load, mem_used, mem_total = map(float, result.strip().split(","))
                    except Exception:
                        gpu_load = 0
                        mem_used = torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)
                        mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 * 1024)

                    mem_util = (mem_used / mem_total) * 100

                    with self.lock:
                        self.data.append({
                            "timestamp": current_time,
                            "gpu_id": gpu_id,
                            "gpu_load": gpu_load,
                            "memory_used_mb": mem_used,
                            "memory_total_mb": mem_total,
                            "memory_util_percent": mem_util,
                        })
                time.sleep(self.interval)

        except Exception as e:
            print(f"[GPUMonitor] Monitoring failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.data:
                self._save_data()

    def stop(self):
        self.stop_event.set()
        self.join(timeout=5)
        print("[GPUMonitor] Stopped")

    def _save_data(self):
        if not self.data:
            return
        df = pd.DataFrame(self.data)
        idx = df.groupby(["timestamp", "gpu_id"])["gpu_load"].idxmax()
        df = df.loc[idx]
        excel_file = os.path.join(self.result_dir, "gpu_monitor.xlsx")
        df.to_excel(excel_file, index=False)
        print(f"[GPUMonitor] Saved GPU metrics to {excel_file}")

    def get_gpu_utilization(self, gpu_id, start_time=None, end_time=None):
        with self.lock:
            if len(self.data) < 2:
                return None
            df = pd.DataFrame(self.data)
            idx = df.groupby(["timestamp", "gpu_id"])["gpu_load"].idxmax()
            df = df.loc[idx]
            gpu_data = df[df["gpu_id"] == gpu_id]
            if len(gpu_data) < 2:
                return None
            if start_time is not None:
                gpu_data = gpu_data[gpu_data["timestamp"] >= start_time]
            if end_time is not None:
                gpu_data = gpu_data[gpu_data["timestamp"] <= end_time]
            if len(gpu_data) < 2:
                return None

            q1 = gpu_data["gpu_load"].quantile(0.25)
            q3 = gpu_data["gpu_load"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered = gpu_data[
                (gpu_data["gpu_load"] >= lower_bound) & (gpu_data["gpu_load"] <= upper_bound)
            ]
            return filtered["gpu_load"].mean() if len(filtered) > 0 else gpu_data["gpu_load"].mean()

    def get_recent_utilization(self, gpu_id, T):
        with self.lock:
            if len(self.data) < 1:
                return None
            df = pd.DataFrame(self.data)
            gpu_data = df[df["gpu_id"] == gpu_id]
            if len(gpu_data) < T:
                return None
            recent = gpu_data.sort_values(by="timestamp", ascending=False).head(T)
            return recent["gpu_load"].mean() if len(recent) >= 1 else None
