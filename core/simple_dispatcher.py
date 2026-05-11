
import threading
import time

from common.state import ReplicaState
from core.dispatcher import BaseDispatcher


class SubflowDispatcher(BaseDispatcher):

    def __init__(self, replicas, ddl=0.8, result_dir="./output"):
        super().__init__(replicas, ddl)
        self.request_ddl = ddl
        self.result_dir = result_dir

        self.model_params = {}
        self.queue_times = []
        self.avg_queue_time = 0.0
        self.queue_time_window = 50

        self.num_subflows = len(replicas)
        self.subflow_configs = []
        self.min_batch_size = 1
        self.adjustment_interval = 5
        self.poll_interval = 0.01
        self.low_queue_ratio = 0.2
        self.high_queue_ratio = 0.6
        self.max_batch_step = 1
        self.cooldown_sec = 3
        self._last_light_switch = time.time() + 60
        self._cooldown_mutex = threading.Lock()

        for replica in self.replicas:
            initial_batch = max(self.min_batch_size, int(replica.infer_batch_size))
            self.subflow_configs.append({
                "replica_id": replica.replica_id,
                "current_mean": 0.0,
                "std_interval": 0.0,
                "batch_size": initial_batch,
                "train_batch_size": None,
                "max_batch_size": initial_batch,
                "priority": replica.performance_score,
                "satisfaction_history": [],
                "avg_satisfaction_rate": 0.0,
                "history_window_size": 20,
                "a": None,
                "b": None,
                "a2": None,
            })

        self.subflow_threads = []
        self.adjustment_thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._last_light_switch = time.time() + 30
        print("[SimpleDispatcher] Started")

        self.subflow_threads = []
        for i in range(self.num_subflows):
            worker = threading.Thread(target=self._subflow_worker, args=(i,), daemon=True)
            worker.start()
            self.subflow_threads.append(worker)

        self.adjustment_thread = threading.Thread(target=self._config_adjustment_worker, daemon=True)
        self.adjustment_thread.start()

    def stop(self):
        self.running = False
        for thread in self.subflow_threads:
            thread.join(timeout=1)
        if self.adjustment_thread:
            self.adjustment_thread.join(timeout=1)
        print("[SimpleDispatcher] Stopped")

    def _is_serving_state(self, replica_state):
        return replica_state in (ReplicaState.SERVING, ReplicaState.IDLE, ReplicaState.COMBINED)

    def _clamp_batch_size(self, requested, max_batch):
        return max(self.min_batch_size, min(int(requested), int(max_batch)))

    def _record_satisfaction(self, config, actual_batch_size, target_batch_size):
        satisfaction = actual_batch_size / target_batch_size if target_batch_size > 0 else 0.0
        config["satisfaction_history"].append(satisfaction)
        if len(config["satisfaction_history"]) > config["history_window_size"]:
            config["satisfaction_history"].pop(0)
        if config["satisfaction_history"]:
            config["avg_satisfaction_rate"] = (
                sum(config["satisfaction_history"]) / len(config["satisfaction_history"])
            )

    def _maybe_switch_to_idle(self, replica, config):
        if replica.check_state() != ReplicaState.SERVING:
            return

        with self.queue_lock:
            backlog = len(self.request_queue)

        if backlog > 0 or self.avg_queue_time > self.low_queue_ratio * self.request_ddl:
            return

        with self._cooldown_mutex:
            now = time.time()
            if now - self._last_light_switch <= self.cooldown_sec:
                return

            should_switch = replica.state_manager.should_switch_to_idle(
                replica=replica,
                replicas=self.replicas,
                gpu_monitor=getattr(self, "gpu_monitor", None),
                process_time=max(0.01, self.avg_queue_time),
                ddl=float(self.request_ddl),
            )
            if should_switch:
                replica.set_state(ReplicaState.IDLE)
                self._last_light_switch = now

    def _dispatch_requests(self, subflow_id, target_batch_size):
        replica = self.replicas[subflow_id]
        dispatch_batch = []
        pending = []

        with self.queue_lock:
            while len(pending) < target_batch_size and self.request_queue:
                pending.append(self.request_queue.popleft())

        for idx, request in enumerate(pending):
            success = replica.add_request(request)
            queue_time = time.time() - request["timestamp"]
            if success:
                dispatch_batch.append(request)
                self._update_avg_queue_time(queue_time)
                self.metrics.append({
                    "request_id": request["request_id"],
                    "replica_id": replica.replica_id,
                    "timestamp": time.time(),
                    "queue_time": queue_time,
                })
            else:
                with self.queue_lock:
                    for remain in reversed(pending[idx + 1:]):
                        self.request_queue.appendleft(remain)
                    self.request_queue.appendleft(request)
                break

        return len(dispatch_batch)

    def _config_adjustment_worker(self):
        while self.running:
            time.sleep(self.adjustment_interval)
            try:
                with self.queue_lock:
                    backlog = len(self.request_queue)

                active_indices = [
                    idx for idx, replica in enumerate(self.replicas)
                    if self._is_serving_state(replica.check_state())
                ]
                if not active_indices:
                    continue

                backlog_per_replica = backlog / max(1, len(active_indices))
                queue_ratio = self.avg_queue_time / max(self.request_ddl, 1e-6)

                for idx in active_indices:
                    cfg = self.subflow_configs[idx]
                    current = self._clamp_batch_size(cfg["batch_size"], cfg["max_batch_size"])
                    max_batch = max(self.min_batch_size, int(cfg["max_batch_size"]))
                    target = current

                    if queue_ratio >= self.high_queue_ratio:
                        target = self.min_batch_size
                    elif backlog_per_replica >= current and cfg["avg_satisfaction_rate"] >= 0.8:
                        target = min(max_batch, current + self.max_batch_step)
                    elif backlog_per_replica < max(1, current // 2):
                        target = max(self.min_batch_size, current - self.max_batch_step)
                    elif queue_ratio <= self.low_queue_ratio and backlog_per_replica == 0:
                        target = self.min_batch_size

                    cfg["batch_size"] = target
                    self.replicas[idx].set_infer_batch_size(target)

                active_batches = [self.subflow_configs[i]["batch_size"] for i in active_indices]
                print(
                    f"[SimpleDispatcher]  batch | queue={self.avg_queue_time:.4f}s "
                    f"| backlog={backlog} | active_batch={active_batches}"
                )
            except Exception as exc:
                print(f"[SimpleDispatcher] Model fitting failed: {exc}")

    def _subflow_worker(self, subflow_id):
        config = self.subflow_configs[subflow_id]
        replica = self.replicas[subflow_id]

        while self.running:
            try:
                replica_state = replica.check_state()
                if not self._is_serving_state(replica_state):
                    time.sleep(self.poll_interval)
                    continue

                max_batch = max(self.min_batch_size, int(config["max_batch_size"]))
                target_batch_size = self._clamp_batch_size(config["batch_size"], max_batch)
                actual_batch_size = self._dispatch_requests(subflow_id, target_batch_size)
                self._record_satisfaction(config, actual_batch_size, target_batch_size)

                if actual_batch_size == 0:
                    self._maybe_switch_to_idle(replica, config)
                    time.sleep(self.poll_interval)
                    continue

                self._maybe_switch_to_idle(replica, config)
            except Exception as exc:
                print(f"[SimpleDispatcher] Subflow {subflow_id} failed: {exc}")
                time.sleep(0.5)

    def update_subflow_config(self, replica_id, batch_size=None, max_batch_size=None,
                              train_batch_size=None, mean_interval=None,
                              std_interval=None, para=None):
        for idx, cfg in enumerate(self.subflow_configs):
            if cfg["replica_id"] != replica_id:
                continue

            if max_batch_size is not None:
                cfg["max_batch_size"] = max(self.min_batch_size, int(max_batch_size))
            if batch_size is not None:
                cfg["batch_size"] = self._clamp_batch_size(batch_size, cfg["max_batch_size"])
                self.replicas[idx].set_infer_batch_size(cfg["batch_size"])
            if train_batch_size is not None:
                cfg["train_batch_size"] = train_batch_size
            if mean_interval is not None:
                cfg["current_mean"] = mean_interval
            if std_interval is not None:
                cfg["std_interval"] = std_interval
            if para and para[0] is not None:
                cfg["a"] = para[0]
            if para and para[1] is not None:
                cfg["a2"] = para[1]
            if para and para[2] is not None:
                cfg["b"] = para[2]
            return True
        return False

    def get_subflow_stats(self, replica_id=None):
        if replica_id is not None:
            for cfg in self.subflow_configs:
                if cfg["replica_id"] == replica_id:
                    return cfg
            return None
        return self.subflow_configs

    def get_avg_queue_time(self):
        return self.avg_queue_time
