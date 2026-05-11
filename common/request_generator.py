import threading
import time
import random
import numpy as np
from collections import deque
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz


class RequestGenerator:
    def __init__(self, inference_dataset, dispatcher, concurrent_requests,
                 mean_interval=1.0, std_dev=0.2, ddl=0.8, max_requests=80000,
                 request_trace=None, start_date=None, duration_time=None,
                 scale_up=1, pattern="trace"):
        self.inference_dataset = inference_dataset
        self.dispatcher = dispatcher
        self.pattern = pattern

        self.mean_interval = mean_interval
        self.std_dev = std_dev
        self.ddl = ddl
        self.max_requests = max_requests

        self.request_trace = request_trace
        self.start_date = start_date
        self.duration_time = duration_time
        self.scale_up = scale_up
        self.trace_timestamps = None

        self.generator_thread = None
        self.running = False
        self.concurrent_requests = concurrent_requests
        self.request_count = 0
        self.metrics = []
        self.lock = threading.Lock()

    def load_trace_range(self):
        try:
            start_datetime = datetime.strptime(self.start_date, "%Y-%m-%d %H:%M:%S")
            start_datetime = pytz.UTC.localize(start_datetime)
            end_datetime = start_datetime + timedelta(seconds=self.duration_time)
            print(f"[RequestGenerator] Trace window: {start_datetime} to {end_datetime}")

            columns = pd.read_csv(self.request_trace, nrows=0).columns.tolist()
            if "TIMESTAMP" not in columns:
                raise ValueError("Trace CSV must contain a TIMESTAMP column")

            chunk_size = 10000
            filtered_data = []
            for chunk in pd.read_csv(self.request_trace, chunksize=chunk_size):
                chunk["datetime"] = pd.to_datetime(chunk["TIMESTAMP"], format="ISO8601")
                mask = (chunk["datetime"] >= start_datetime) & (chunk["datetime"] <= end_datetime)
                filtered_chunk = chunk[mask]
                if not filtered_chunk.empty:
                    filtered_data.append(filtered_chunk)

            if filtered_data:
                filtered_df = pd.concat(filtered_data).sort_values("datetime")
                self.trace_timestamps = filtered_df["datetime"].tolist()
                print(f"[RequestGenerator] Loaded trace events: {len(self.trace_timestamps)}")
                if self.trace_timestamps:
                    print(f"[RequestGenerator] Trace range: {self.trace_timestamps[0]} to {self.trace_timestamps[-1]}")
                return True
            else:
                print("[RequestGenerator] No trace events found in the selected window")
                return False
        except Exception as e:
            print(f"[RequestGenerator] Trace loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self, duration=None, trace_date=None, trace_duration=None):
        if self.generator_thread is not None and self.generator_thread.is_alive():
            print("[RequestGenerator] Request generator is already running")
            return

        self.running = True
        if self.trace_timestamps is not None:
            self.generator_thread = threading.Thread(target=self._generate_from_trace)
            print(f"[RequestGenerator] Starting trace replay; scaled events={self.scale_up * len(self.trace_timestamps)}")
        else:
            self.generator_thread = threading.Thread(target=self._generate_loop, args=(duration,))
            print(f"[RequestGenerator] Starting fixed generator; interval={self.mean_interval}, concurrency={self.concurrent_requests}")

        self.generator_thread.daemon = True
        self.generator_thread.start()

    def _generate_from_trace(self):
        try:
            if not self.trace_timestamps:
                print("[RequestGenerator] No trace events to replay")
                return

            random.seed(42)
            intervals = [
                (self.trace_timestamps[i] - self.trace_timestamps[i - 1]).total_seconds()
                for i in range(1, len(self.trace_timestamps))
            ]

            dataset_size = len(self.inference_dataset)
            for i, timestamp in enumerate(self.trace_timestamps):
                if not self.running:
                    break
                if i > 0:
                    time.sleep(intervals[i - 1])

                if self.scale_up >= 1:
                    indices = [random.randint(0, dataset_size - 1) for _ in range(int(self.scale_up))]
                    samples = [self.inference_dataset[idx] for idx in indices]
                else:
                    if random.random() < self.scale_up:
                        idx = random.randint(0, dataset_size - 1)
                        indices = [idx]
                        samples = [self.inference_dataset[idx]]
                    else:
                        continue

                request_ids = []
                request_timestamps = []
                for sample in samples:
                    request = {
                        "instruction": sample["instruction"],
                        "input": sample.get("input", ""),
                        "output": sample["output"],
                        "ddl": self.ddl,
                    }
                    rid = self.dispatcher.add_request(request)
                    request_ids.append(rid)
                    request_timestamps.append(request["timestamp"])

                with self.lock:
                    self.request_count += len(samples)
                    for j, rid in enumerate(request_ids):
                        self.metrics.append({
                            "request_id": rid,
                            "timestamp": request_timestamps[j],
                            "dataset_idx": indices[j],
                            "trace_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        })

            print(f"[RequestGenerator] Trace replay finished; generated={self.request_count}")
        except Exception as e:
            print(f"[RequestGenerator] Generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _generate_loop(self, duration):
        start_time = time.time()
        dataset_size = len(self.inference_dataset)

        try:
            while self.running:
                if duration is not None and time.time() - start_time > duration:
                    print(f"[RequestGenerator] Fixed generation duration reached: {duration}s")
                    break

                indices = [random.randint(0, dataset_size - 1) for _ in range(self.concurrent_requests)]
                samples = [self.inference_dataset[idx] for idx in indices]

                request_ids = []
                request_timestamps = []
                for i, sample in enumerate(samples):
                    request = {
                        "instruction": sample["instruction"],
                        "input": sample.get("input", ""),
                        "output": sample["output"],
                        "ddl": self.ddl,
                    }
                    rid = self.dispatcher.add_request(request)
                    request_ids.append(rid)
                    request_timestamps.append(request["timestamp"])

                with self.lock:
                    self.request_count += len(samples)
                    for i, rid in enumerate(request_ids):
                        self.metrics.append({
                            "request_id": rid,
                            "timestamp": request_timestamps[i],
                            "dataset_idx": indices[i],
                        })

                time.sleep(self.mean_interval)

        except Exception as e:
            print(f"[RequestGenerator] Fixed generation failed: {e}")
            import traceback
            traceback.print_exc()

    def get_metrics(self):
        with self.lock:
            return {"request_count": self.request_count, "metrics": self.metrics}

    def adjust_rate(self, new_mean_interval, new_std_dev=None):
        with self.lock:
            self.mean_interval = new_mean_interval
            if new_std_dev is not None:
                self.std_dev = new_std_dev
            print(f"[RequestGenerator] Updated mean interval: {self.mean_interval}")

    def stop(self):
        self.running = False
        if self.generator_thread and self.generator_thread.is_alive():
            self.generator_thread.join(timeout=2)
            print("[RequestGenerator] Stopped")
