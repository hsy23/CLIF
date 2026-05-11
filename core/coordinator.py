
import threading
import time
from statistics import mean

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


class BatchSizeOptimizer:

    def __init__(self, process_delay, result_dir="./output"):
        self.result_dir = result_dir
        self.lock = threading.Lock()
        self.training_data = []
        self.inference_data = []
        self.batch_size_history = {}
        self.process_delay = process_delay
        self.min_process_delay = 0.0
        self.infer_batch_size_history = {}
        self.inference_offsets = {}

        self.min_inference_samples_per_group = 3
        self.max_inference_samples = 5000
        self.history_rounds = 3
        self.min_training_fit_points = 4
        self.min_inference_fit_points = 4
        self.min_unique_train_batch_sizes = 2
        self.min_unique_infer_batch_sizes = 2
        self.coefficient_epsilon = 1e-6


    def _prune_samples_by_round_locked(self, samples):
        round_ids = sorted({s.get("round_id") for s in samples if s.get("round_id") is not None})
        if len(round_ids) <= self.history_rounds:
            return samples
        keep_rounds = set(round_ids[-self.history_rounds:])
        return [s for s in samples if s.get("round_id") is None or s.get("round_id") in keep_rounds]

    def _sample_overlaps_time_window(self, metric, time_window):
        if not time_window:
            return True
        train_start_time, train_end_time = time_window
        if train_start_time is None or train_end_time is None:
            return True

        finished_time = metric.get("finished_time")
        process_time = metric.get("process_time")
        if finished_time is None or process_time is None:
            return False

        service_end_time = finished_time
        service_start_time = finished_time - process_time
        return service_end_time >= train_start_time and service_start_time <= train_end_time

    def _get_recent_batch_size(self, replica_id, kind, default):
        history = self.batch_size_history.get(f"{replica_id}_{kind}", [])
        if history:
            return history[-1]
        return default

    def _build_simple_infer_candidates(self, replica_count, min_infer_batch, max_infer_batch):
        mid_infer_batch = int(round((min_infer_batch + max_infer_batch) / 2))
        if replica_count <= 1:
            return [mid_infer_batch]
        if replica_count == 2:
            return sorted(set([min_infer_batch, max_infer_batch]))
        return sorted(set([min_infer_batch, mid_infer_batch, max_infer_batch]))

    def _build_simple_fallback_results(self, replica_ids, min_train_batch, min_infer_batch, max_infer_batch, reason):
        results = {}
        sorted_replica_ids = sorted(replica_ids)
        infer_candidates = self._build_simple_infer_candidates(
            len(sorted_replica_ids), min_infer_batch, max_infer_batch
        )
        print(
            f"[BatchSizeOptimizer] Fallback triggered: {reason}; "
            f"infer_candidates={infer_candidates}"
        )
        for idx, replica_id in enumerate(sorted_replica_ids):
            train_batch_size = self._get_recent_batch_size(replica_id, "train", min_train_batch)
            if len(sorted_replica_ids) == 1:
                infer_batch_size = self._get_recent_batch_size(
                    replica_id, "infer", infer_candidates[0]
                )
            else:
                infer_batch_size = infer_candidates[idx % len(infer_candidates)]
            results[replica_id] = (train_batch_size, infer_batch_size)
        return results

    def _summarize_training_groups(self, data):
        group_counts = {}
        unique_train_batch_sizes = set()
        unique_infer_batch_sizes = set()
        round_ids = set()

        for item in data:
            key = (item["train_batch_size"], item["infer_batch_size"])
            group_counts[key] = group_counts.get(key, 0) + 1
            unique_train_batch_sizes.add(item["train_batch_size"])
            unique_infer_batch_sizes.add(item["infer_batch_size"])
            if item.get("round_id") is not None:
                round_ids.add(item["round_id"])

        return {
            "total_samples": len(data),
            "unique_train_batch_sizes": len(unique_train_batch_sizes),
            "unique_infer_batch_sizes": len(unique_infer_batch_sizes),
            "group_counts": group_counts,
            "round_ids": sorted(round_ids),
        }

    def _summarize_inference_groups(self, data):
        group_counts = {}
        unique_train_batch_sizes = set()
        unique_infer_batch_sizes = set()
        round_ids = set()

        for item in data:
            key = (item["train_batch_size"], item["infer_batch_size"])
            group_counts[key] = group_counts.get(key, 0) + 1
            unique_train_batch_sizes.add(item["train_batch_size"])
            unique_infer_batch_sizes.add(item["infer_batch_size"])
            if item.get("round_id") is not None:
                round_ids.add(item["round_id"])

        return {
            "total_samples": len(data),
            "unique_train_batch_sizes": len(unique_train_batch_sizes),
            "unique_infer_batch_sizes": len(unique_infer_batch_sizes),
            "group_counts": group_counts,
            "round_ids": sorted(round_ids),
        }


    def collect_training_data(
        self,
        replica_id,
        train_batch_size,
        infer_batch_size,
        avg_iteration_time,
        avg_loss_decrease,
        gradient_noise,
        train_start_time=None,
        train_end_time=None,
        round_id=None,
        infer_batch_source="configured_fallback",
    ):
        with self.lock:
            self.training_data.append(
                {
                    "replica_id": replica_id,
                    "train_batch_size": train_batch_size,
                    "infer_batch_size": infer_batch_size,
                    "avg_iteration_time": avg_iteration_time,
                    "avg_loss_decrease": avg_loss_decrease,
                    "gradient_noise": gradient_noise,
                    "train_start_time": train_start_time,
                    "train_end_time": train_end_time,
                    "round_id": round_id,
                    "infer_batch_source": infer_batch_source,
                }
            )
            self.training_data = self._prune_samples_by_round_locked(self.training_data)
            self.batch_size_history.setdefault(f"{replica_id}_train", []).append(train_batch_size)
            self.batch_size_history.setdefault(f"{replica_id}_infer", []).append(infer_batch_size)

    def collect_inference_data(
        self,
        replica_id,
        train_batch_size,
        inference_metrics,
        time_window=None,
        round_id=None,
    ):
        if not inference_metrics:
            return {
                "observed_infer_batch_size": None,
                "accepted_samples": 0,
                "rejected_samples": 0,
            }

        offset = self.inference_offsets.get(replica_id, 0)
        new_metrics = inference_metrics[offset:]
        self.inference_offsets[replica_id] = len(inference_metrics)
        if not new_metrics:
            print(f"[BatchSizeOptimizer] Replica {replica_id} no new inference metrics to collect")
            return {
                "observed_infer_batch_size": None,
                "accepted_samples": 0,
                "rejected_samples": 0,
            }

        filtered_metrics = [
            metric for metric in new_metrics if self._sample_overlaps_time_window(metric, time_window)
        ]
        accepted_samples = len(filtered_metrics)
        rejected_samples = len(new_metrics) - accepted_samples

        observed_infer_batch_size = None
        if filtered_metrics:
            observed_infer_batch_size = round(
                mean([metric.get("infer_batch_size", 1) for metric in filtered_metrics])
            )
            self.infer_batch_size_history[replica_id] = observed_infer_batch_size

        batch_groups = {}
        for metric in filtered_metrics:
            infer_batch_size = metric.get("infer_batch_size", 1)
            process_time = metric.get("process_time")
            if process_time is None:
                continue
            batch_groups.setdefault(infer_batch_size, []).append(process_time)

        group_counts = {infer_batch_size: len(times) for infer_batch_size, times in batch_groups.items()}
        print(
            f"[BatchSizeOptimizer] Replica {replica_id} inference collection: "
            f"new_samples={len(new_metrics)}, accepted_samples={accepted_samples}, "
            f"rejected_samples={rejected_samples}, train_bs={train_batch_size}, "
            f"observed_infer_bs={observed_infer_batch_size}, group_counts={group_counts}"
        )

        if batch_groups:
            with self.lock:
                for infer_batch_size, times in batch_groups.items():
                    for process_time in times:
                        self.inference_data.append(
                            {
                                "replica_id": replica_id,
                                "train_batch_size": train_batch_size,
                                "infer_batch_size": infer_batch_size,
                                "avg_process_time": process_time,
                                "sample_count": 1,
                                "timestamp": time.time(),
                                "round_id": round_id,
                            }
                        )
                self.inference_data = self._prune_samples_by_round_locked(self.inference_data)
                if len(self.inference_data) > self.max_inference_samples:
                    self.inference_data = self.inference_data[-self.max_inference_samples :]

        return {
            "observed_infer_batch_size": observed_infer_batch_size,
            "accepted_samples": accepted_samples,
            "rejected_samples": rejected_samples,
        }


    def _fit_training_model(self, replica_id=None):
        with self.lock:
            data = self.training_data
            if replica_id is not None:
                data = [item for item in data if item["replica_id"] == replica_id]

            if len(data) < self.min_training_fit_points:
                return None, None, None, None

            summary = self._summarize_training_groups(data)
            if (
                summary["unique_train_batch_sizes"] < self.min_unique_train_batch_sizes
                or summary["unique_infer_batch_sizes"] < self.min_unique_infer_batch_sizes
                or len(summary["group_counts"]) < self.min_training_fit_points
            ):
                return None, None, None, None

            groups = {}
            for item in data:
                key = (item["train_batch_size"], item["infer_batch_size"])
                groups.setdefault(key, []).append(item["avg_iteration_time"])

            X = []
            y = []
            for (train_batch_size, infer_batch_size), times in groups.items():
                X.append([train_batch_size, infer_batch_size])
                y.append(sum(times) / len(times))

            if len(X) < self.min_training_fit_points:
                return None, None, None, None

            try:
                reg = LinearRegression().fit(np.array(X), np.array(y))
                a1, b1 = reg.coef_
                c1 = reg.intercept_
                return max(0, a1), max(0, b1), max(0, c1), reg.score(np.array(X), np.array(y))
            except Exception:
                return None, None, None, None

    def _fit_inference_model(self, replica_id=None):
        with self.lock:
            data = self.inference_data
            if replica_id is not None:
                data = [item for item in data if item["replica_id"] == replica_id]

            if len(data) < self.min_inference_fit_points:
                return None, None, None, None

            raw_groups = {}
            for item in data:
                key = (item["infer_batch_size"], item["train_batch_size"])
                raw_groups.setdefault(key, []).append(item["avg_process_time"])

            fit_groups = {}
            for key, times in raw_groups.items():
                if len(times) >= self.min_inference_samples_per_group:
                    fit_groups[key] = times

            unique_infer_batch_sizes = {infer_batch_size for infer_batch_size, _ in fit_groups}
            unique_train_batch_sizes = {train_batch_size for _, train_batch_size in fit_groups}
            if (
                len(fit_groups) < self.min_inference_fit_points
                or len(unique_infer_batch_sizes) < self.min_unique_infer_batch_sizes
                or len(unique_train_batch_sizes) < self.min_unique_train_batch_sizes
            ):
                return None, None, None, None

            X = []
            y = []
            for (infer_batch_size, train_batch_size), times in fit_groups.items():
                X.append([infer_batch_size, train_batch_size])
                y.append(sum(times) / len(times))

            try:
                reg = LinearRegression().fit(np.array(X), np.array(y))
                a2, b2 = reg.coef_
                c2 = reg.intercept_
                return max(0, a2), max(0, b2), max(0, c2), reg.score(np.array(X), np.array(y))
            except Exception:
                return None, None, None, None


    def _get_max_x2(self, x1, a2, b2, c2):
        if a2 is None or a2 <= self.coefficient_epsilon:
            return None
        return (self.process_delay - b2 * x1 - c2) / a2

    def _objective_function_x1_only(self, x1, a1, b1, c1, p, l, b0, a2, b2, c2, min_ib, max_ib):
        x1 = float(np.atleast_1d(x1)[0])
        a_factor = 80
        max_x2 = self._get_max_x2(x1, a2, b2, c2)
        x2 = min_ib if max_x2 is None else min(max_ib, max(min_ib, max_x2))
        if x1 <= 0 or a1 * x1 + b1 * x2 + c1 <= 0 or p * l + x1 <= 0:
            return 1e10
        term1 = x1 / (a1 * x1 + b1 * x2 + c1)
        term2 = (a_factor * p * l + b0) / (a_factor * p * l + x1)
        return -(term1 * term2)

    def _optimize_batch_sizes(
        self,
        replica_id,
        train_fit_parameter,
        infer_fit_parameter,
        min_train_batch=1,
        max_train_batch=32,
        min_infer_batch=1,
        max_infer_batch=32,
    ):
        recent = None
        with self.lock:
            training_data = [item for item in self.training_data if item["replica_id"] == replica_id]
            if training_data:
                recent = training_data[-1]
        if not recent:
            return min_train_batch, min_infer_batch

        x0 = recent.get("train_batch_size", min_train_batch)
        p = recent.get("gradient_noise", 0.1)
        l = recent.get("avg_loss_decrease", 0.01)
        print(f"[BatchSizeOptimizer] Replica {replica_id} p={p:.4f}, l={l:.4f}")

        a1, b1, c1 = train_fit_parameter
        a2, b2, c2 = infer_fit_parameter
        train_history = self.batch_size_history.get(f"{replica_id}_train", [])
        b0 = train_history[0] if train_history else min_train_batch

        try:
            if (
                a1 is None
                or b1 is None
                or c1 is None
                or a2 is None
                or b2 is None
                or c2 is None
                or a2 <= self.coefficient_epsilon
            ):
                return x0, recent.get("infer_batch_size", min_infer_batch)

            result = minimize(
                lambda candidate_x1: self._objective_function_x1_only(
                    candidate_x1,
                    a1,
                    b1,
                    c1,
                    p,
                    l,
                    b0,
                    a2,
                    b2,
                    c2,
                    min_infer_batch,
                    max_infer_batch,
                ),
                x0=x0,
                method="L-BFGS-B",
                bounds=[(min_train_batch, max_train_batch)],
                options={"maxiter": 100},
            )
            if result.success:
                opt_tb = max(min_train_batch, min(max_train_batch, round(result.x[0])))
                max_x2 = self._get_max_x2(opt_tb, a2, b2, c2)
                if max_x2 is None:
                    return x0, recent.get("infer_batch_size", min_infer_batch)
                opt_ib = max(min_infer_batch, min(max_infer_batch, round(max_x2)))
                obj = self._objective_function_x1_only(
                    opt_tb,
                    a1,
                    b1,
                    c1,
                    p,
                    l,
                    b0,
                    a2,
                    b2,
                    c2,
                    min_infer_batch,
                    max_infer_batch,
                )
                print(f"[BatchSizeOptimizer] Replica {replica_id} goodput={-obj:.4f}")
                return opt_tb, opt_ib
            return x0, recent.get("infer_batch_size", min_infer_batch)
        except Exception as exc:
            print(f"[BatchSizeOptimizer] Optimization exception: {exc}")
            return x0, recent.get("infer_batch_size", min_infer_batch)

    def optimize_batch_sizes(self, replica_ids, min_train_batch=2, max_train_batch=16, min_infer_batch=2, max_infer_batch=20):
        def _fmt(value):
            return "None" if value is None else f"{value:.4f}"

        results = {}
        try:
            with self.lock:
                training_summary = self._summarize_training_groups(self.training_data)
                inference_summary = self._summarize_inference_groups(self.inference_data)

            print(
                "[BatchSizeOptimizer] Training pool summary: "
                f"rounds={training_summary['round_ids']}, samples={training_summary['total_samples']}, "
                f"unique_train_bs={training_summary['unique_train_batch_sizes']}, "
                f"unique_infer_bs={training_summary['unique_infer_batch_sizes']}, "
                f"group_counts={training_summary['group_counts']}"
            )
            print(
                "[BatchSizeOptimizer] Inference pool summary: "
                f"rounds={inference_summary['round_ids']}, samples={inference_summary['total_samples']}, "
                f"unique_train_bs={inference_summary['unique_train_batch_sizes']}, "
                f"unique_infer_bs={inference_summary['unique_infer_batch_sizes']}, "
                f"group_counts={inference_summary['group_counts']}"
            )

            a1, b1, c1, r2t = self._fit_training_model()
            a2, b2, c2, r2i = self._fit_inference_model()

            print(f"[BatchSizeOptimizer] Training model: y1={_fmt(a1)}*x1+{_fmt(b1)}*x2+{_fmt(c1)}, R2={r2t}")
            print(f"[BatchSizeOptimizer] Inference model: y2={_fmt(a2)}*x2+{_fmt(b2)}*x1+{_fmt(c2)}, R2={r2i}")

            training_identifiable = all(value is not None for value in (a1, b1, c1))
            inference_identifiable = all(value is not None for value in (a2, b2, c2)) and a2 > self.coefficient_epsilon

            if not training_identifiable or not inference_identifiable:
                reasons = []
                if not training_identifiable:
                    reasons.append("training_model_unidentifiable")
                if not inference_identifiable:
                    reasons.append("inference_model_unidentifiable")
                results = self._build_simple_fallback_results(
                    replica_ids,
                    min_train_batch,
                    min_infer_batch,
                    max_infer_batch,
                    ",".join(reasons),
                )
                for replica_id, (train_batch_size, infer_batch_size) in results.items():
                    with self.lock:
                        self.batch_size_history.setdefault(f"{replica_id}_train", []).append(train_batch_size)
                        self.batch_size_history.setdefault(f"{replica_id}_infer", []).append(infer_batch_size)
                return results, [None, None, None]

            for replica_id in replica_ids:
                train_batch_size, infer_batch_size = self._optimize_batch_sizes(
                    replica_id,
                    [a1, b1, c1],
                    [a2, b2, c2],
                    min_train_batch,
                    max_train_batch,
                    min_infer_batch,
                    max_infer_batch,
                )
                results[replica_id] = (train_batch_size, infer_batch_size)
                with self.lock:
                    self.batch_size_history.setdefault(f"{replica_id}_train", []).append(train_batch_size)
                    self.batch_size_history.setdefault(f"{replica_id}_infer", []).append(infer_batch_size)
            return results, [a2, b2, c2]
        except Exception as exc:
            print(f"[BatchSizeOptimizer] Optimization failed: {exc}; using simple fallback")
            results = self._build_simple_fallback_results(
                replica_ids,
                min_train_batch,
                min_infer_batch,
                max_infer_batch,
                "optimizer_exception",
            )
            for replica_id, (train_batch_size, infer_batch_size) in results.items():
                with self.lock:
                    self.batch_size_history.setdefault(f"{replica_id}_train", []).append(train_batch_size)
                    self.batch_size_history.setdefault(f"{replica_id}_infer", []).append(infer_batch_size)
            return results, [None, None, None]

    def get_initial_batch_sizes(self, replica_ids, min_train_batch=1, max_train_batch=5, min_infer_batch=1, max_infer_batch=5):
        results = {}
        train_batch_sizes = np.random.choice(
            range(min_train_batch, max_train_batch + 1),
            size=len(replica_ids),
            replace=len(replica_ids) > max_train_batch - min_train_batch + 1,
        )
        infer_batch_sizes = np.random.choice(
            range(min_infer_batch, max_infer_batch + 1),
            size=len(replica_ids),
            replace=len(replica_ids) > max_infer_batch - min_infer_batch + 1,
        )
        for idx, replica_id in enumerate(replica_ids):
            train_batch_size, infer_batch_size = int(train_batch_sizes[idx]), int(infer_batch_sizes[idx])
            results[replica_id] = (train_batch_size, infer_batch_size)
            with self.lock:
                self.batch_size_history.setdefault(f"{replica_id}_train", []).append(train_batch_size)
                self.batch_size_history.setdefault(f"{replica_id}_infer", []).append(infer_batch_size)
        return results

    def update_process_delay(self, process_delay, min_process_delay=None):
        if min_process_delay is not None:
            self.min_process_delay = max(0.0, min_process_delay)
        self.process_delay = max(process_delay, self.min_process_delay)


class Coordinator:

    def __init__(self, min_train_batch, max_train_batch, min_infer_batch, max_infer_batch, ddl, result_dir="./output"):
        self.min_train_batch = min_train_batch
        self.max_train_batch = max_train_batch
        self.min_infer_batch = min_infer_batch
        self.max_infer_batch = max_infer_batch
        self.batch_optimizer = BatchSizeOptimizer(ddl, result_dir)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.replica_metrics = {}

    def collect_training_metrics(self, replica_id, metrics, round_id=None):
        if not metrics:
            return
        with self.lock:
            self.replica_metrics.setdefault(replica_id, {"training": [], "inference": []})
            self.replica_metrics[replica_id]["training"].append(metrics)
        self.batch_optimizer.collect_training_data(
            replica_id=replica_id,
            train_batch_size=metrics.get("train_batch_size", 1),
            infer_batch_size=metrics.get("infer_batch_size", 1) or 1,
            avg_iteration_time=metrics.get("avg_iteration_time", 0.1),
            avg_loss_decrease=metrics.get("avg_loss_decrease", 0.01),
            gradient_noise=metrics.get("gradient_noise", 0.1),
            train_start_time=metrics.get("train_start_time"),
            train_end_time=metrics.get("train_end_time"),
            round_id=round_id if round_id is not None else metrics.get("round_id"),
            infer_batch_source=metrics.get("infer_batch_source", "configured_fallback"),
        )

    def collect_inference_metrics(
        self,
        replica_id,
        metrics,
        train_batch_size=None,
        train_time_window=None,
        round_id=None,
    ):
        if not metrics:
            return {
                "observed_infer_batch_size": None,
                "accepted_samples": 0,
                "rejected_samples": 0,
            }
        with self.lock:
            self.replica_metrics.setdefault(replica_id, {"training": [], "inference": []})
            self.replica_metrics[replica_id]["inference"].extend(metrics)

        if train_batch_size is None:
            with self.lock:
                training_metrics = self.replica_metrics.get(replica_id, {}).get("training", [])
                if training_metrics:
                    recent = max(training_metrics, key=lambda metric: metric.get("timestamp", 0))
                    train_batch_size = recent.get("train_batch_size", 1)
                    start_time = recent.get("train_start_time")
                    end_time = recent.get("train_end_time")
                    if start_time and end_time:
                        train_time_window = (start_time, end_time)

        return self.batch_optimizer.collect_inference_data(
            replica_id=replica_id,
            train_batch_size=train_batch_size,
            inference_metrics=metrics,
            time_window=train_time_window,
            round_id=round_id,
        )

    def optimize_batch_sizes(self, replica_ids):
        return self.batch_optimizer.optimize_batch_sizes(
            replica_ids,
            self.min_train_batch,
            self.max_train_batch,
            self.min_infer_batch,
            self.max_infer_batch,
        )

    def get_initial_batch_sizes(self, replica_ids):
        return self.batch_optimizer.get_initial_batch_sizes(
            replica_ids,
            self.min_train_batch,
            self.max_train_batch,
            self.min_infer_batch,
            self.max_infer_batch,
        )

    def update_process_delay(self, process_delay, min_process_delay=None):
        self.batch_optimizer.update_process_delay(process_delay, min_process_delay)
