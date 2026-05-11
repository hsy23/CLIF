
import inspect
import threading
import time

import torch

from common.state import ReplicaState


def average_state_dicts(state_dicts):
    avg = {}
    for key in state_dicts[0].keys():
        avg[key] = sum(state_dict[key].cpu().float() for state_dict in state_dicts) / len(state_dicts)
    return avg


class FederatedLearningLauncher:

    def __init__(
        self,
        replicas,
        request_dispatcher,
        coordinator,
        state_manager,
        local_steps=100,
        rounds=3,
        baseline="CLIF",
        swap_strategy="fixed",
        swap_interval=50,
        swap_loss_delta=0.05,
        min_participants=1,
    ):
        self.replicas = replicas
        self.request_dispatcher = request_dispatcher
        self.coordinator = coordinator
        self.state_manager = state_manager
        self.local_steps = local_steps
        self.rounds = rounds
        self.baseline = baseline
        self.swap_strategy = swap_strategy
        self.swap_interval = swap_interval
        self.swap_loss_delta = swap_loss_delta
        self.min_participants = min_participants

        self.stop_event = threading.Event()
        self.fl_thread = None
        self.lock = threading.Lock()
        self.round_metrics = []


    def start_fl(self):
        if self.fl_thread is not None and self.fl_thread.is_alive():
            print("[FL] Federated learning is already running")
            return
        self.stop_event.clear()
        target = self._fl_process_c if self.coordinator else self._fl_process
        self.fl_thread = threading.Thread(target=target, daemon=True)
        self.fl_thread.start()

    def stop_fl(self):
        self.stop_event.set()
        if self.fl_thread:
            self.fl_thread.join(timeout=10)
            print("[FL] Federated learning stopped")
        return True


    def _fl_process_c(self):
        eps = 1e-3
        round_index = 0
        check_period = 10

        while round_index < self.rounds:
            while True:
                if self.stop_event.is_set():
                    break
                idle_ids = self.state_manager.get_replicas_by_state(["IDLE"])
                if not idle_ids:
                    print("[FL] Available replicas: [], waiting for IDLE replicas")
                    time.sleep(check_period)
                    continue

                selected_ids, pressure = self._select_replicas_by_service_pressure(idle_ids)
                skipped_ids = [rid for rid in idle_ids if rid not in set(selected_ids)]
                if skipped_ids:
                    self._release_idle_replicas_to_serving(skipped_ids)
                if selected_ids:
                    break
                print(
                    f"[FL] Service pressure is {pressure['level']}; "
                    f"postponing this FL round, released IDLE replicas to SERVING: {skipped_ids}"
                )
                time.sleep(check_period)

            if self.stop_event.is_set():
                break

            round_id = round_index + 1
            print(f"[FL] Start round {round_id}/{self.rounds}, participants: {selected_ids}")
            selected_replicas = [replica for replica in self.replicas if replica.replica_id in selected_ids]
            for replica in selected_replicas:
                if replica.check_state() != ReplicaState.COMBINED:
                    self.state_manager.reset_idle_counter(replica.replica_id)
                    replica.set_state(ReplicaState.COMBINED)

            round_start_time = time.time()

            process_time = self.request_dispatcher.get_available_service_time(selected_ids)
            min_process_time = self.request_dispatcher.get_min_service_time_budget(selected_ids)
            self.coordinator.update_process_delay(process_time, min_process_time)

            if round_index == 0:
                batch_sizes = self.coordinator.get_initial_batch_sizes(selected_ids)
                first_round = True
                fit_parameters = [None, None, None]
            else:
                batch_sizes, fit_parameters = self.coordinator.optimize_batch_sizes(selected_ids)
                first_round = False

            for replica_id in selected_ids:
                self.request_dispatcher.update_subflow_config(
                    replica_id,
                    batch_sizes[replica_id][-1],
                    batch_sizes[replica_id][-1],
                    train_batch_size=batch_sizes[replica_id][0],
                    para=fit_parameters,
                )
            print(f"[FL] Batch sizes for this round: {batch_sizes}")

            train_results = self._train_participants(selected_replicas, batch_sizes, first_round)

            for replica in selected_replicas:
                replica_id = replica.replica_id
                train_metric = train_results.get(replica_id)
                if not train_metric or replica_id not in batch_sizes:
                    continue

                train_batch_size, configured_infer_batch_size = batch_sizes[replica_id]
                serve_metrics = replica.get_metrics().get("serve_metrics", [])
                train_start_time = train_metric.get("train_start_time")
                train_end_time = train_metric.get("train_end_time")
                observed_info = {
                    "observed_infer_batch_size": None,
                    "accepted_samples": 0,
                    "rejected_samples": 0,
                }

                if serve_metrics:
                    observed_info = self.coordinator.collect_inference_metrics(
                        replica_id,
                        serve_metrics,
                        train_batch_size,
                        (train_start_time, train_end_time),
                        round_id=round_id,
                    )

                observed_infer_batch_size = observed_info.get("observed_infer_batch_size")
                train_metric["configured_infer_batch_size"] = configured_infer_batch_size
                train_metric["observed_infer_batch_size"] = observed_infer_batch_size
                train_metric["infer_batch_size"] = (
                    observed_infer_batch_size
                    if observed_infer_batch_size is not None
                    else configured_infer_batch_size
                )
                train_metric["infer_batch_source"] = (
                    "observed" if observed_infer_batch_size is not None else "configured_fallback"
                )
                train_metric["round_id"] = round_id

            for replica_id, metrics in train_results.items():
                if metrics:
                    self.coordinator.collect_training_metrics(replica_id, metrics, round_id=round_id)

            global_state = self._aggregate_models(selected_replicas, train_results)
            if global_state:
                print(f"[FL] Aggregation completed for round {round_id}")
                for replica in self.replicas:
                    if replica.replica_id not in selected_ids:
                        replica.download_model(global_state)

            round_time = time.time() - round_start_time
            average_metrics = self._calculate_average_metrics(train_results)
            self._record_round(
                round_index,
                selected_ids,
                batch_sizes,
                round_time,
                average_metrics,
                selected_replicas,
                pressure=pressure,
            )
            print(
                f"[FL] Round {round_id} finished in {round_time:.2f}s, "
                f"loss={average_metrics.get('train_loss', 0)}"
            )
            round_index += 1

            for replica in selected_replicas:
                replica.set_state(ReplicaState.IDLE)


        print(f"[FL] Federated learning completed, total rounds={len(self.round_metrics)}")
        with self.lock:
            self.fl_thread = None
        for replica in self.replicas:
            if replica.check_state() in (ReplicaState.IDLE, ReplicaState.COMBINED):
                replica.set_state(ReplicaState.SERVING)


    def _fl_process(self):
        round_index = 0
        check_period = 10

        while round_index < self.rounds:
            if self.stop_event.is_set():
                break
            print(f"[FL] Start round {round_index + 1}/{self.rounds}")

            while True:
                idle_ids = self.state_manager.get_replicas_by_state(["IDLE"])
                if not idle_ids:
                    print("[FL] Available replicas: [], waiting for IDLE replicas")
                    time.sleep(check_period)
                    continue

                selected_ids, pressure = self._select_replicas_by_service_pressure(idle_ids)
                skipped_ids = [rid for rid in idle_ids if rid not in set(selected_ids)]
                if skipped_ids:
                    self._release_idle_replicas_to_serving(skipped_ids)
                if selected_ids:
                    break
                if self.stop_event.is_set():
                    break
                print(
                    f"[FL] Service pressure is {pressure['level']}; "
                    f"postponing this FL round, released IDLE replicas to SERVING: {skipped_ids}"
                )
                time.sleep(check_period)
            if self.stop_event.is_set():
                break

            selected_replicas = [replica for replica in self.replicas if replica.replica_id in selected_ids]

            round_start_time = time.time()
            batch_sizes = {
                replica.replica_id: (replica.train_batch_size, replica.infer_batch_size)
                for replica in selected_replicas
            }
            print(f"[FL] Batch sizes for this round: {batch_sizes}")

            for replica in selected_replicas:
                self.state_manager.reset_idle_counter(replica.replica_id)
                replica.set_state(ReplicaState.COMBINED)

            train_results = self._train_participants(selected_replicas, batch_sizes)
            global_state = self._aggregate_models(selected_replicas, train_results)
            if global_state:
                print(f"[FL] Aggregation completed for round {round_index + 1}")

            round_time = time.time() - round_start_time
            average_metrics = self._calculate_average_metrics(train_results)
            self._record_round(
                round_index,
                [replica.replica_id for replica in selected_replicas],
                batch_sizes,
                round_time,
                average_metrics,
                selected_replicas,
                pressure=pressure,
            )
            print(
                f"[FL] Round {round_index + 1} finished in {round_time:.2f}s, "
                f"loss={average_metrics.get('train_loss', 0)}"
            )

            for replica in selected_replicas:
                replica.set_state(ReplicaState.IDLE)
            round_index += 1

        print(f"[FL] Federated learning completed, total rounds={len(self.round_metrics)}")
        with self.lock:
            self.fl_thread = None
        for replica in self.replicas:
            if replica.check_state() in (ReplicaState.IDLE, ReplicaState.COMBINED):
                replica.set_state(ReplicaState.SERVING)


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

    @staticmethod
    def _fmt_metric(value):
        if value is None:
            return "None"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _get_replica(self, replica_id):
        for replica in self.replicas:
            if replica.replica_id == replica_id:
                return replica
        return None

    def _release_idle_replicas_to_serving(self, replica_ids):
        for replica_id in replica_ids:
            replica = self._get_replica(replica_id)
            if replica is None:
                continue
            if replica.check_state() == ReplicaState.IDLE:
                replica.set_state(ReplicaState.SERVING)

    def _replica_pressure_score(self, replica, window=20):
        queue_len = replica.get_queue_length() if hasattr(replica, "get_queue_length") else 0
        metrics = replica.get_metrics().get("serve_metrics", [])[-window:]
        service_times = []
        pre_service_waits = []
        for metric in metrics:
            service_time = metric.get("total_service_time")
            process_time = metric.get("process_time")
            try:
                service_time = float(service_time)
            except (TypeError, ValueError):
                service_time = None
            try:
                process_time = float(process_time)
            except (TypeError, ValueError):
                process_time = None
            if service_time is not None:
                service_times.append(service_time)
            if service_time is not None and process_time is not None:
                pre_service_waits.append(max(0.0, service_time - process_time))

        ddl = max(1e-6, float(getattr(self.request_dispatcher, "request_ddl", self.request_dispatcher.ddl)))
        service_p90 = self._percentile(service_times, 90) or 0.0
        wait_p90 = self._percentile(pre_service_waits, 90) or 0.0
        return queue_len + service_p90 / ddl + wait_p90 / ddl

    def _select_replicas_by_service_pressure(self, idle_ids):
        if hasattr(self.request_dispatcher, "get_service_pressure"):
            pressure = self.request_dispatcher.get_service_pressure()
        else:
            pressure = {
                "level": "low",
                "combined_ratio": 1.0,
                "recent_success_rate": None,
                "dispatch_queue_p90": 0.0,
                "dispatch_backlog": 0,
                "service_time_p90": None,
                "reasons": ["pressure_api_unavailable"],
            }

        ratio = max(0.0, min(1.0, float(pressure.get("combined_ratio", 1.0))))
        target_count = int(len(idle_ids) * ratio)
        if ratio >= 1.0:
            target_count = len(idle_ids)
        target_count = max(0, min(len(idle_ids), target_count))

        scored_ids = []
        for rid in idle_ids:
            replica = self._get_replica(rid)
            score = self._replica_pressure_score(replica) if replica is not None else float("inf")
            scored_ids.append((score, rid))
        ranked_ids = [rid for _, rid in sorted(scored_ids)]
        selected_ids = ranked_ids[:target_count]
        skipped_ids = ranked_ids[target_count:]

        print(
            "[FL] Service pressure gate: "
            f"level={pressure.get('level')}, ratio={ratio:.2f}, "
            f"idle={idle_ids}, selected={selected_ids}, skipped={skipped_ids}, "
            f"success_rate={self._fmt_metric(pressure.get('recent_success_rate'))}, "
            f"dispatch_p90={self._fmt_metric(pressure.get('dispatch_queue_p90'))}, "
            f"service_p90={self._fmt_metric(pressure.get('service_time_p90'))}, "
            f"backlog={pressure.get('dispatch_backlog')}, "
            f"reasons={pressure.get('reasons')}"
        )
        return selected_ids, pressure

    def _record_round(
        self,
        round_index,
        participant_ids,
        batch_sizes,
        round_time,
        average_metrics,
        selected_replicas,
        pressure=None,
    ):
        metric = {
            "round": round_index + 1,
            "participants": len(selected_replicas),
            "participant_ids": participant_ids,
            "batch_sizes": batch_sizes,
            "round_time_sec": round_time,
            "avg_train_loss": average_metrics.get("train_loss", 0),
            "avg_loss_decrease": average_metrics.get("avg_loss_decrease", 0),
            "avg_iteration_time": average_metrics.get("avg_iteration_time", 0),
            "avg_gradient_noise": average_metrics.get("gradient_noise", 0),
            "timestamp": time.time(),
        }
        if pressure:
            metric.update({
                "pressure_level": pressure.get("level"),
                "pressure_combined_ratio": pressure.get("combined_ratio"),
                "pressure_success_rate": pressure.get("recent_success_rate"),
                "pressure_dispatch_queue_p90": pressure.get("dispatch_queue_p90"),
                "pressure_service_time_p90": pressure.get("service_time_p90"),
                "pressure_dispatch_backlog": pressure.get("dispatch_backlog"),
                "pressure_reasons": ";".join(pressure.get("reasons", [])),
            })
        with self.lock:
            self.round_metrics.append(metric)

    def _train_participants(self, selected_replicas, batch_sizes, first_round=None):
        results = {}
        threads = []
        for replica in selected_replicas:
            train_batch_size, infer_batch_size = batch_sizes.get(replica.replica_id, (1, 1))
            if first_round:
                max_steps = max(1, self.local_steps // 2)
            else:
                max_steps = self.local_steps
            replica.set_infer_batch_size(infer_batch_size)
            thread = threading.Thread(
                target=self._train_replica,
                args=(replica, replica.replica_id, results, train_batch_size, max_steps),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return results

    def _train_replica(self, replica, index, results, batch_size=None, max_steps=None):
        try:
            train_kwargs = {"max_steps": max_steps, "batch_size": batch_size}
            train_params = inspect.signature(replica.local_train).parameters
            if "swap_strategy" in train_params:
                train_kwargs.update(
                    {
                        "swap_strategy": self.swap_strategy,
                        "swap_interval": self.swap_interval,
                        "swap_loss_delta": self.swap_loss_delta,
                    }
                )
            metrics = replica.local_train(**train_kwargs)
            results[index] = metrics
        except Exception as exc:
            print(f"[FL] Replica {replica.replica_id} training failed: {exc}")
            results[index] = None

    def _aggregate_models(self, replicas, train_results):
        if not replicas:
            return None
        try:
            state_dicts = []
            for replica in replicas:
                result = train_results.get(replica.replica_id)
                if result is not None:
                    try:
                        state_dicts.append(replica.get_state_dict())
                        if result.get("export_adapter_name"):
                            print(
                                f"[FL] Replica {replica.replica_id} export adapter: "
                                f"{result['export_adapter_name']}"
                            )
                    except Exception as exc:
                        print(f"[FL] Failed to read state from replica {replica.replica_id}: {exc}")
            if not state_dicts:
                print("[FL] No model states available for aggregation, skipping this round")
                return None

            aggregated = average_state_dicts(state_dicts)
            threads = []
            for replica in replicas:
                if hasattr(replica, "download_model"):
                    thread = threading.Thread(target=replica.download_model, args=(aggregated,))
                else:
                    thread = threading.Thread(target=replica.update_model, args=(aggregated,))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            print("[FL] All models updated")
            return aggregated
        except Exception as exc:
            print(f"[FL] Aggregation failed: {exc}")
            return None

    def _calculate_average_metrics(self, train_results):
        valid_metrics = [metrics for metrics in train_results.values() if metrics is not None]
        if not valid_metrics:
            return {}
        average_metrics = {}
        for key in ("train_loss", "avg_loss_decrease", "avg_iteration_time", "gradient_noise", "wall_time_sec"):
            values = [metrics.get(key, 0) for metrics in valid_metrics]
            average_metrics[key] = sum(values) / len(values) if values else 0
        return average_metrics

    def get_metrics(self):
        with self.lock:
            return {"round_metrics": self.round_metrics.copy()}
