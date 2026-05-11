
import threading
import time
import uuid
from collections import deque
from sklearn.linear_model import LinearRegression
from common.state import ReplicaState



class BaseDispatcher:

    def __init__(self, replicas, ddl):
        self.replicas = replicas
        self.request_queue = deque()
        self.queue_lock = threading.Lock()
        self.running = False
        self.metrics = []
        self.ddl = ddl


    def add_request(self, request):
        if "request_id" not in request:
            request["request_id"] = str(uuid.uuid4())
        request["timestamp"] = time.time()
        with self.queue_lock:
            self.request_queue.append(request)
        return request["request_id"]


    def get_metrics(self):
        return self.metrics

    def get_all_states(self):
        return [r.check_state() for r in self.replicas]

    @staticmethod
    def _to_float(value):
        try:
            if value is None:
                return None
            value = float(value)
            if value != value:
                return None
            return value
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _percentile(values, percentile):
        clean_values = sorted(v for v in values if v is not None)
        if not clean_values:
            return None
        if len(clean_values) == 1:
            return clean_values[0]
        rank = (len(clean_values) - 1) * percentile / 100.0
        lower = int(rank)
        upper = min(lower + 1, len(clean_values) - 1)
        weight = rank - lower
        return clean_values[lower] * (1 - weight) + clean_values[upper] * weight

    def get_service_pressure(self, replica_ids=None, window=50, min_served_requests=20):
        replica_id_set = set(replica_ids) if replica_ids is not None else None
        serve_metrics = []
        for replica in self.replicas:
            if replica_id_set is not None and replica.replica_id not in replica_id_set:
                continue
            metrics = replica.get_metrics().get("serve_metrics", [])
            if metrics:
                serve_metrics.extend(metrics[-window:])

        served_requests = 0
        successful_requests = 0
        service_times = []
        process_times = []
        pre_service_waits = []
        for metric in serve_metrics:
            batch_size = self._to_float(metric.get("infer_batch_size"))
            success_batch = self._to_float(metric.get("s_batch"))
            if batch_size is not None:
                served_requests += max(0, int(batch_size))
            if success_batch is not None:
                successful_requests += max(0, int(success_batch))

            service_time = self._to_float(metric.get("total_service_time"))
            process_time = self._to_float(metric.get("process_time"))
            if service_time is not None:
                service_times.append(service_time)
            if process_time is not None:
                process_times.append(process_time)
            if service_time is not None and process_time is not None:
                pre_service_waits.append(max(0.0, service_time - process_time))

        recent_success_rate = (
            successful_requests / served_requests if served_requests > 0 else None
        )

        dispatch_metrics = list(self.metrics[-window:])
        dispatch_queue_times = [
            self._to_float(metric.get("queue_time")) for metric in dispatch_metrics
        ]
        dispatch_queue_times = [v for v in dispatch_queue_times if v is not None]
        dispatch_queue_mean = (
            sum(dispatch_queue_times) / len(dispatch_queue_times)
            if dispatch_queue_times else 0.0
        )
        dispatch_queue_p90 = self._percentile(dispatch_queue_times, 90) or 0.0

        with self.queue_lock:
            dispatch_backlog = len(self.request_queue)

        ddl = max(1e-6, float(getattr(self, "request_ddl", self.ddl)))
        infer_batches = []
        for replica in self.replicas:
            try:
                infer_batches.append(max(1, int(getattr(replica, "infer_batch_size", 1))))
            except (TypeError, ValueError):
                infer_batches.append(1)
        max_infer_batch = max(infer_batches or [1])
        backlog_high = 2 * max(1, len(self.replicas)) * max_infer_batch
        sample_sufficient = served_requests >= min_served_requests

        service_time_mean = sum(service_times) / len(service_times) if service_times else None
        service_time_p90 = self._percentile(service_times, 90)
        process_time_p90 = self._percentile(process_times, 90)
        pre_service_wait_p90 = self._percentile(pre_service_waits, 90)

        high_reasons = []
        medium_reasons = []

        if dispatch_queue_p90 > ddl:
            high_reasons.append(f"dispatch_queue_p90>{ddl:.3f}s")
        elif dispatch_queue_p90 > 0.3 * ddl:
            medium_reasons.append(f"dispatch_queue_p90>{0.3 * ddl:.3f}s")

        if dispatch_backlog > backlog_high:
            high_reasons.append(f"dispatch_backlog>{backlog_high}")
        elif dispatch_backlog > 0:
            medium_reasons.append("dispatch_backlog>0")

        if sample_sufficient:
            if recent_success_rate is not None:
                if recent_success_rate < 0.5:
                    high_reasons.append("success_rate<0.50")
                elif recent_success_rate < 0.85:
                    medium_reasons.append("success_rate<0.85")
            if service_time_p90 is not None:
                if service_time_p90 > 1.25 * ddl:
                    high_reasons.append(f"service_time_p90>{1.25 * ddl:.3f}s")
                elif service_time_p90 > 0.9 * ddl:
                    medium_reasons.append(f"service_time_p90>{0.9 * ddl:.3f}s")
        elif served_requests > 0:
            medium_reasons.append("insufficient_recent_served_samples")

        if high_reasons:
            level = "high"
            combined_ratio = 0.0
            reasons = high_reasons
        elif medium_reasons:
            level = "medium"
            combined_ratio = 0.5
            reasons = medium_reasons
        else:
            level = "low"
            combined_ratio = 1.0
            reasons = ["healthy"]

        return {
            "level": level,
            "combined_ratio": combined_ratio,
            "recent_success_rate": recent_success_rate,
            "served_requests": served_requests,
            "successful_requests": successful_requests,
            "serve_batches": len(serve_metrics),
            "sample_sufficient": sample_sufficient,
            "dispatch_queue_mean": dispatch_queue_mean,
            "dispatch_queue_p90": dispatch_queue_p90,
            "dispatch_backlog": dispatch_backlog,
            "service_time_mean": service_time_mean,
            "service_time_p90": service_time_p90,
            "process_time_p90": process_time_p90,
            "pre_service_wait_p90": pre_service_wait_p90,
            "reasons": reasons,
        }


    def _update_avg_queue_time(self, new_queue_time):
        self.queue_times.append(new_queue_time)
        if len(self.queue_times) > self.queue_time_window:
            self.queue_times.pop(0)
        if self.queue_times:
            self.avg_queue_time = sum(self.queue_times) / len(self.queue_times)

    def _collect_inference_metrics(self):
        metrics_by_replica = {}
        for replica in self.replicas:
            if replica.check_state() not in (ReplicaState.SERVING, ReplicaState.IDLE):
                continue
            replica_metrics = replica.get_metrics()
            sm = replica_metrics.get("serve_metrics", [])
            if sm:
                offset = self.metric_offsets.get(replica.replica_id, 0)
                new_metrics = sm[offset:]
                self.metric_offsets[replica.replica_id] = len(sm)
                if new_metrics:
                    metrics_by_replica[replica.replica_id] = new_metrics
        return metrics_by_replica

    def _summarize_fit_metrics(self, metrics):
        batch_groups = {}
        for metric in metrics:
            bs = metric.get("infer_batch_size", 1)
            batch_groups[bs] = batch_groups.get(bs, 0) + 1
        return {
            "total_samples": len(metrics),
            "unique_batch_sizes": len(batch_groups),
            "batch_counts": batch_groups,
        }

    def _build_exploration_candidates(self, initial_batch_size):
        upper = max(self.min_infer_batch_size, min(int(initial_batch_size), self.min_infer_batch_size + 2))
        return list(range(self.min_infer_batch_size, upper + 1))

    def _exploration_active(self, config):
        if config.get("exploration_completed", False):
            return False
        if time.time() - self._exploration_start_time > self.exploration_duration_sec:
            config["exploration_completed"] = True
            return False
        if self.avg_queue_time > self.exploration_backlog_ratio_limit * self.request_ddl:
            return False
        for candidate in config["exploration_candidates"]:
            if config["exploration_counts"].get(candidate, 0) < self.exploration_min_batches_per_bs:
                return True
        config["exploration_completed"] = True
        return False

    def _choose_exploration_batch_size(self, config):
        for candidate in config["exploration_candidates"]:
            if config["exploration_counts"].get(candidate, 0) < self.exploration_min_batches_per_bs:
                return candidate
        return config["batch_size"]

    def _record_exploration_sample(self, config, actual_batch_size):
        if actual_batch_size <= 0:
            return
        if actual_batch_size in config["exploration_candidates"]:
            counts = config["exploration_counts"]
            counts[actual_batch_size] = counts.get(actual_batch_size, 0) + 1
        if not self._exploration_active(config):
            config["exploration_completed"] = True

    def _fit_inference_model(self, metrics, default_a=0.2, default_b=0.1):
        if not metrics or len(metrics) < self.fit_min_total_samples:
            return default_a, default_b, None

        batch_groups = {}
        for m in metrics:
            bs = m.get("infer_batch_size", 1)
            batch_groups.setdefault(bs, []).append(m.get("process_time", 0))

        X, y = [], []
        for bs, times in batch_groups.items():
            if len(times) >= self.fit_min_samples_per_bs:
                times.sort()
                trimmed = times[1:-1] if len(times) > 2 else times
                X.append([bs])
                y.append(sum(trimmed) / len(trimmed))

        if len(X) < 2:
            return default_a, default_b, None

        try:
            reg = LinearRegression().fit(X, y)
            a = max(0.01, reg.coef_[0])
            b = max(0.01, reg.intercept_)
            return a, b, reg.score(X, y)
        except Exception as e:
            print(f"[Dispatcher] Inference-model fitting failed: {e}")
            return default_a, default_b, None


    def start(self):
        raise NotImplementedError

    def stop(self):
        self.running = False



class SubflowDispatcher(BaseDispatcher):
    def __init__(self, replicas, ddl=0.8, result_dir="./output"):
        super().__init__(replicas, ddl)
        self.request_ddl = ddl
        self.result_dir = result_dir
        default_min_infer_batch = min((int(replica.infer_batch_size) for replica in replicas), default=4)
        self.min_infer_batch_size = max(4, default_min_infer_batch)
        self.fit_min_total_samples = 5
        self.fit_min_samples_per_bs = 5
        self.exploration_duration_sec = 60
        self.exploration_min_batches_per_bs = 6
        self.exploration_backlog_ratio_limit = 0.6
        self._exploration_start_time = time.time()
        self.metric_offsets = {}

        self.model_params = {}
        self.queue_times = []
        self.avg_queue_time = 0.0
        self.queue_time_window = 50
        self.inference_metrics_for_fitting = []

        self.cooldown_sec = 3
        self._last_light_switch = time.time() + 60
        self._cooldown_mutex = threading.Lock()

        self.num_subflows = len(replicas)
        self.subflow_configs = []
        for replica in self.replicas:
            self.subflow_configs.append({
                "replica_id": replica.replica_id,
                "current_mean": 0.5,
                "std_interval": 0.01,
                "batch_size": replica.infer_batch_size,
                "train_batch_size": None,
                "max_batch_size": replica.infer_batch_size,
                "priority": replica.performance_score,
                "satisfaction_history": [],
                "avg_satisfaction_rate": 0.0,
                "history_window_size": 20,
                "a": 0.01, "b": 0.01, "a2": None,
                "exploration_candidates": self._build_exploration_candidates(replica.infer_batch_size),
                "exploration_counts": {},
                "exploration_completed": False,
            })

        self.subflow_threads = []
        self.adjustment_thread = None
        self.model_fitting_thread = None

        self.small_adjustment_interval = 30
        self.large_adjustment_interval = 100


    def _calculate_batch_size_from_model(self, available_time, a, b):
        if a <= 0:
            return self.min_infer_batch_size
        return max(self.min_infer_batch_size, int((available_time - b) / a))

    def _calculate_interval_from_batch_size(self, batch_size, a, b):
        return a * batch_size + b

    def _calculate_interval_from_batch_size_combined(self, batch_size, train_batch_size, a, b, a2):
        return a * batch_size + a2 * train_batch_size + b

    def _estimate_batch_process_time(self, replica_state, config, batch_size):
        if batch_size <= 0:
            return 0.01
        if replica_state == ReplicaState.COMBINED and config["a2"] is not None and config["train_batch_size"] is not None:
            return max(0.01, self._calculate_interval_from_batch_size_combined(
                batch_size, config["train_batch_size"], config["a"], config["b"], config["a2"]))
        return max(0.01, self._calculate_interval_from_batch_size(batch_size, config["a"], config["b"]))

    def _should_drop_batch(self, batch, replica_state, config):
        if not batch:
            return False
        now = time.time()
        estimated_process_time = self._estimate_batch_process_time(replica_state, config, len(batch))
        return all(
            (req.get("ddl", self.request_ddl) - (now - req.get("timestamp", now))) <= estimated_process_time
            for req in batch
        )

    def _record_dropped_batch(self, replica_id, batch, reason):
        now = time.time()
        for request in batch:
            request_ts = request.get("timestamp", now)
            request_ddl = request.get("ddl", self.request_ddl)
            self.metrics.append({
                "request_id": request.get("request_id"),
                "replica_id": replica_id,
                "timestamp": now,
                "queue_time": max(0.0, now - request_ts),
                "dropped": True,
                "drop_reason": reason,
                "remaining_ddl": request_ddl - (now - request_ts),
            })

    def estimate_recent_pre_service_wait(self, replica_ids=None, window=50):
        samples = []
        replica_id_set = set(replica_ids) if replica_ids is not None else None
        for replica in self.replicas:
            if replica_id_set is not None and replica.replica_id not in replica_id_set:
                continue
            serve_metrics = replica.get_metrics().get("serve_metrics", [])
            if not serve_metrics:
                continue
            for metric in serve_metrics[-window:]:
                total_service_time = metric.get("total_service_time")
                process_time = metric.get("process_time")
                if total_service_time is None or process_time is None:
                    continue
                samples.append(max(0.0, total_service_time - process_time))
        if not samples:
            return self.avg_queue_time
        if len(samples) > window:
            samples = samples[-window:]
        return sum(samples) / len(samples)

    def get_min_service_time_budget(self, replica_ids=None):
        budgets = []
        replica_id_set = set(replica_ids) if replica_ids is not None else None
        for cfg in self.subflow_configs:
            if replica_id_set is not None and cfg["replica_id"] not in replica_id_set:
                continue
            budgets.append(self._estimate_batch_process_time(
                ReplicaState.SERVING, cfg, self.min_infer_batch_size))
        if not budgets:
            return max(0.05, min(self.request_ddl, 0.1 * self.request_ddl))
        return max(0.05, min(self.request_ddl, sum(budgets) / len(budgets)))

    def get_available_service_time(self, replica_ids=None):
        pre_service_wait = self.estimate_recent_pre_service_wait(replica_ids)
        min_service_budget = self.get_min_service_time_budget(replica_ids)
        return max(min_service_budget, self.request_ddl - pre_service_wait)


    def start(self):
        if self.running:
            return
        self._exploration_start_time = time.time()
        self._last_light_switch = time.time() + 30
        self.running = True
        print("[Dispatcher] Started")

        self.subflow_threads = []
        for i in range(self.num_subflows):
            t = threading.Thread(target=self._subflow_worker, args=(i,), daemon=True)
            t.start()
            self.subflow_threads.append(t)

        self.adjustment_thread = threading.Thread(target=self._config_adjustment_worker, daemon=True)
        self.adjustment_thread.start()

        self.model_fitting_thread = threading.Thread(target=self._model_fitting_worker, daemon=True)
        self.model_fitting_thread.start()

    def stop(self):
        self.running = False
        for t in self.subflow_threads:
            t.join(timeout=1)
        if self.adjustment_thread:
            self.adjustment_thread.join(timeout=1)
        if self.model_fitting_thread:
            self.model_fitting_thread.join(timeout=1)
        print("[Dispatcher] Stopped")


    def _model_fitting_worker(self):
        time.sleep(20)
        while self.running:
            try:
                print("[Dispatcher] Fitting inference model")
                for mlist in self._collect_inference_metrics().values():
                    self.inference_metrics_for_fitting.extend(mlist)
                if len(self.inference_metrics_for_fitting) > 1000:
                    self.inference_metrics_for_fitting = self.inference_metrics_for_fitting[-1000:]

                fit_summary = self._summarize_fit_metrics(self.inference_metrics_for_fitting)
                a, b, r2 = self._fit_inference_model(self.inference_metrics_for_fitting)
                print(f"[Dispatcher] Fitted inference model: y = {a:.4f}*x + {b:.4f}, R2={r2}")

                print(
                    f"[Dispatcher] Fit data summary: samples={fit_summary['total_samples']}, "
                    f"unique_bs={fit_summary['unique_batch_sizes']}, batch_counts={fit_summary['batch_counts']}"
                )
                if r2 is None:
                    print("[Dispatcher] Inference fit insufficient data, using default or latest fitted parameters")
                estimated_pre_service_wait = self.estimate_recent_pre_service_wait()
                estimated_queue_time = estimated_pre_service_wait


                for replica in self.replicas:
                    rid = replica.replica_id
                    rstate = replica.check_state()
                    available_service_time = self.get_available_service_time([rid])
                    for cfg in self.subflow_configs:
                        if cfg["replica_id"] == rid:
                            cfg["a"] = a
                            cfg["b"] = b
                            if rstate != ReplicaState.COMBINED:
                                bs = self._calculate_batch_size_from_model(available_service_time, a, b)
                                cfg["max_batch_size"] = bs
                                interval = self._calculate_interval_from_batch_size(bs, a, b)
                                cfg["current_mean"] = interval
                                print(
                                    f"[Dispatcher] Replica {rid} ({rstate}): "
                                    f"bs={bs}, interval={interval:.4f}s, available={available_service_time:.4f}s"
                                )

                time.sleep(self.large_adjustment_interval)
            except Exception as e:
                print(f"[Dispatcher] Model fitting loop failed: {e}")
                time.sleep(10)


    def _config_adjustment_worker(self):
        time.sleep(self.large_adjustment_interval)
        while self.running:
            time.sleep(self.small_adjustment_interval)
            try:
                states = [r.check_state() for r in self.replicas]
                active_indices = [
                    idx for idx in range(len(states))
                    if not self._exploration_active(self.subflow_configs[idx])
                ]
                if not active_indices:
                    continue

                priorities = []
                u_asrs = []
                for i in active_indices:
                    base_priority = self.replicas[i].performance_score
                    cfg = self.subflow_configs[i]
                    unsatisfaction_rate = 1 - cfg["avg_satisfaction_rate"]
                    u_asrs.append(unsatisfaction_rate)
                    adjusted = base_priority * (1 + unsatisfaction_rate)
                    cfg["priority"] = adjusted
                    priorities.append(adjusted)

                total_priority = sum(priorities)
                if total_priority == 0:
                    continue
                norm = [p / total_priority for p in priorities]
                pdev = max(norm) - min(norm)

                total_cap = sum(self.subflow_configs[i]["max_batch_size"] for i in active_indices)
                total_batch = sum(self.subflow_configs[i]["batch_size"] for i in active_indices)

                if (pdev < 0.05 or max(u_asrs) <= 0.3) and abs(total_batch - total_cap) <= 0.2 * total_cap:
                    continue

                min_bs = 2
                remaining = total_cap - len(active_indices) * min_bs
                ideal = [min_bs] * len(active_indices)
                if remaining > 0:
                    for j, p in enumerate(norm):
                        ideal[j] += round(remaining * p)

                for j, idx in enumerate(active_indices):
                    cfg = self.subflow_configs[idx]
                    cur = cfg["batch_size"]
                    target = ideal[j]
                    max_change = max(1, int(cur * 0.5))
                    if target > cur:
                        new = min(target, cur + max_change, cfg["max_batch_size"])
                    else:
                        new = max(min_bs, target, cur - max_change)
                    cfg["batch_size"] = new
                    self.replicas[idx].set_infer_batch_size(new)
                    interval = self._calculate_interval_from_batch_size(new, cfg["a"], cfg["b"])
                    cfg["current_mean"] = interval

                adj_bs = [self.subflow_configs[i]["batch_size"] for i in active_indices]
                print(f"[Adjuster] u_asr={u_asrs} | priorities={[f'{p:.2f}' for p in norm]} | "
                      f"batch: {adj_bs} | ideal: {ideal}")
            except Exception as e:
                print(f"[Adjuster] Adjustment loop failed: {e}")


    def _subflow_worker(self, subflow_id):
        config = self.subflow_configs[subflow_id]
        while self.running:
            try:
                current_mean = config["current_mean"]
                batch_size = config["batch_size"]

                replica_state = self.replicas[subflow_id].check_state()

                valid_batch = []
                dispatch_batch = []
                if self._exploration_active(config):
                    target_batch_size = self._choose_exploration_batch_size(config)
                    if target_batch_size != batch_size:
                        batch_size = target_batch_size
                        config["batch_size"] = target_batch_size
                        self.replicas[subflow_id].set_infer_batch_size(target_batch_size)
                with self.queue_lock:
                    while len(valid_batch) < batch_size:
                        if self.request_queue:
                            valid_batch.append(self.request_queue.popleft())
                        else:
                            break


                queue_time = 0
                if valid_batch:
                    for request in valid_batch:
                        request["dispatch_time"] = time.time()
                        success = self.replicas[subflow_id].add_request(request)
                        queue_time = time.time() - request["timestamp"]
                        if success:
                            dispatch_batch.append(request)
                            self._update_avg_queue_time(queue_time)
                            self.metrics.append({
                                "request_id": request["request_id"],
                                "replica_id": self.replicas[subflow_id].replica_id,
                                "timestamp": time.time(),
                                "queue_time": queue_time,
                            })
                        else:
                            with self.queue_lock:
                                self.request_queue.appendleft(request)

                actual_batch_size = len(dispatch_batch)
                self._record_exploration_sample(config, actual_batch_size)
                satisfaction_rate = actual_batch_size / batch_size if batch_size > 0 else 0
                config["satisfaction_history"].append(satisfaction_rate)
                if len(config["satisfaction_history"]) > config["history_window_size"]:
                    config["satisfaction_history"].pop(0)
                if config["satisfaction_history"]:
                    config["avg_satisfaction_rate"] = (
                        sum(config["satisfaction_history"]) / len(config["satisfaction_history"])
                    )

                replica = self.replicas[subflow_id]
                switch = True
                if replica.check_state() != ReplicaState.SERVING:
                    switch = False
                with self._cooldown_mutex:
                    now = time.time()
                    if switch and now - self._last_light_switch > self.cooldown_sec:
                        switch = replica.state_manager.should_switch_to_idle(
                            replica=replica,
                            replicas=self.replicas,
                            gpu_monitor=getattr(self, "gpu_monitor", None),
                            process_time=config["a"],
                            ddl=float(self.request_ddl),
                        )
                        if switch:
                            replica.set_state(ReplicaState.IDLE)
                            self._last_light_switch = now


                if replica_state in (ReplicaState.SERVING, ReplicaState.IDLE) and config["a"]:
                    current_mean = self._calculate_interval_from_batch_size(
                        actual_batch_size, config["a"], config["b"])
                elif replica_state == ReplicaState.COMBINED and config["a2"]:
                    current_mean = self._calculate_interval_from_batch_size_combined(
                        actual_batch_size, config["train_batch_size"],
                        config["a"], config["b"], config["a2"])

                sleep_time = max(0.01, current_mean)
                if self.avg_queue_time < 0.8 * self.request_ddl:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"[SubFlow {subflow_id}] Dispatch loop failed: {e}")
                time.sleep(0.5)


    def update_subflow_config(self, replica_id, batch_size=None, max_batch_size=None,
                              train_batch_size=None, mean_interval=None,
                              std_interval=None, para=None):
        for cfg in self.subflow_configs:
            if cfg["replica_id"] == replica_id:
                if batch_size is not None:
                    cfg["batch_size"] = batch_size
                if max_batch_size is not None:
                    cfg["max_batch_size"] = max_batch_size
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
