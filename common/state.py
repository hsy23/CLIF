import numpy as np
import threading
import time
from datetime import datetime
from enum import Enum


class ReplicaState(Enum):
    SERVING = "SERVING"
    IDLE = "IDLE"
    TRAINING = "TRAINING"
    COMBINED = "COMBINED"


class StateManagement:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.replica_states = {}
                cls._instance.state_history = []
                cls._instance.run_start_time = time.time()
                cls._instance.state_lock = threading.Lock()
        return cls._instance

    def __init__(self, idle_strategy: str = "slo", idle_timeout: int = 5, ewma_alpha: float = 0.3):
        if not hasattr(self, "state_history"):
            self.state_history = []
        if not hasattr(self, "run_start_time"):
            self.run_start_time = time.time()
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.idle_strategy = idle_strategy
        self.idle_timeout = idle_timeout
        self.ewma_alpha = ewma_alpha

        self.slo_base = 0.9
        self.w1 = 0.5
        self.w2 = 0.5
        self.alpha = 0.1
        self.threshold = 0.8

        if not hasattr(self, "_ewma_util"):
            self._ewma_util = {}
            self._ewma_queue = {}

        if not hasattr(self, "_idle_since"):
            self._idle_since = {}

    @staticmethod
    def _state_value(state):
        return state.value if hasattr(state, "value") else state

    def _record_state_event_locked(self, replica_id, old_state, new_state, event, changed):
        timestamp = time.time()
        self.state_history.append({
            "timestamp": timestamp,
            "relative_time_sec": timestamp - self.run_start_time,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "replica_id": replica_id,
            "old_state": self._state_value(old_state),
            "new_state": self._state_value(new_state),
            "event": event,
            "changed": bool(changed),
        })

    def register_replica(self, replica_id, state):
        with self.state_lock:
            if isinstance(replica_id, list):
                for rid in replica_id:
                    self.replica_states[rid] = state
                    self._record_state_event_locked(
                        rid, old_state=None, new_state=state,
                        event="register", changed=True,
                    )
                return [f"Replica {rid} registered as {state}" for rid in replica_id]
            self.replica_states[replica_id] = state
            self._record_state_event_locked(
                replica_id, old_state=None, new_state=state,
                event="register", changed=True,
            )
            return f"Replica {replica_id} registered as {state}"

    def update_state(self, replica_id, new_state):
        with self.state_lock:
            if isinstance(replica_id, list):
                results = []
                for rid in replica_id:
                    if rid in self.replica_states:
                        old_state = self.replica_states[rid]
                        self.replica_states[rid] = new_state
                        self._record_state_event_locked(
                            rid, old_state=old_state, new_state=new_state,
                            event="update", changed=(old_state != new_state),
                        )
                        results.append(f"Replica {rid} updated to {new_state}")
                    else:
                        results.append(f"Replica {rid} is not registered")
                return results
            if replica_id in self.replica_states:
                old_state = self.replica_states[replica_id]
                self.replica_states[replica_id] = new_state
                self._record_state_event_locked(
                    replica_id, old_state=old_state, new_state=new_state,
                    event="update", changed=(old_state != new_state),
                )
                return f"Replica {replica_id} updated to {new_state}"
            return f"Replica {replica_id} is not registered"

    def get_state(self, replica_id):
        with self.state_lock:
            if isinstance(replica_id, list):
                return {rid: self.replica_states.get(rid) for rid in replica_id}
            return self.replica_states.get(replica_id)

    def get_all_states(self):
        with self.state_lock:
            return self.replica_states.copy()

    def get_state_history(self):
        with self.state_lock:
            return [record.copy() for record in self.state_history]

    def reset_state_history(self):
        with self.state_lock:
            self.state_history = []
            self.run_start_time = time.time()

    def get_replicas_by_state(self, target_state):
        with self.state_lock:
            if isinstance(target_state, (list, tuple)):
                return [
                    rid for rid, state in self.replica_states.items()
                    if state in target_state
                    or (hasattr(state, "value") and state.value in target_state)
                ]
            return [
                rid for rid, state in self.replica_states.items()
                if state == target_state
                or (hasattr(state, "value") and state.value == target_state)
            ]

    def switch_to_idle(self, gpu_monitor, replicas, recent_queue,
                       time_window=10, alpha=0.25):
        all_states = self.get_all_states()
        serving_replicas = [rid for rid, st in all_states.items() if st == ReplicaState.SERVING]

        if not serving_replicas:
            return []

        all_utils = {}
        for rid in serving_replicas:
            util = gpu_monitor.get_recent_utilization(rid, time_window)
            all_utils[rid] = util if util is not None else 50.0

        if not all_utils or not recent_queue:
            return []

        u_threshold = np.quantile(list(all_utils.values()), alpha)
        q_threshold = np.quantile(list(recent_queue.values()), alpha)
        u_threshold_fixed = 25

        switched = []
        for rid in serving_replicas:
            u_current = all_utils[rid]
            q_current = recent_queue.get(rid, 0)
            if u_current < min(u_threshold, u_threshold_fixed):
                self.update_state(rid, ReplicaState.IDLE)
                switched.append(rid)
                print(f"[StateManagement] Replica {rid} SERVING -> IDLE "
                      f"(util={u_current:.2f} < {u_threshold:.2f}, queue={q_current} < {q_threshold})")
        return switched


    def _compute_slo_scores(self, rate_slo: float, wait_time: float, process_time: float, ddl: float):
        if rate_slo < self.slo_base:
            return False, 0.0, 0.0, 0.0
        slo_score = (rate_slo - self.slo_base) / (1.0 - self.slo_base)
        remain = float(ddl) - process_time
        if remain <= 1e-6 or wait_time > remain:
            return False, slo_score, 0.0, 0.0
        wait_score = 1.0 - (wait_time / remain)
        overall = self.w1 * slo_score + self.w2 * wait_score
        return True, slo_score, wait_score, overall

    def should_switch_idle_slo(self, replica, process_time: float, ddl: float):
        if not hasattr(replica, "_slo_recent") or not hasattr(replica, "_wait_recent"):
            return False

        recent_slo = (sum(replica._slo_recent) / len(replica._slo_recent)) if replica._slo_recent else 0.0
        recent_wait = (sum(replica._wait_recent) / len(replica._wait_recent)) if replica._wait_recent else 0.0

        ok_curr, slo_score, wait_score, current_overall = self._compute_slo_scores(
            recent_slo, recent_wait, process_time, ddl
        )
        if not ok_curr:
            return False

        avg_hist_slo = (replica._slo_hist_sum / replica._slo_hist_cnt) if replica._slo_hist_cnt > 0 else 0.0
        avg_hist_wait = (replica._wait_hist_sum / replica._wait_hist_cnt) if replica._wait_hist_cnt > 0 else 0.0

        ok_hist, hist_slo_score, hist_wait_score, _hist_overall = self._compute_slo_scores(
            avg_hist_slo, avg_hist_wait, process_time, ddl
        )
        if not ok_hist:
            return False

        diff_slo = slo_score - hist_slo_score
        diff_wait = wait_score - hist_wait_score

        if diff_slo * diff_wait > 0:
            history_adjust = -abs(self.w1 * diff_slo + self.w2 * diff_wait)
        elif diff_slo < 0:
            history_adjust = self.w1 * diff_slo
        else:
            history_adjust = self.w2 * diff_wait

        selected_len = len(self.get_replicas_by_state(["IDLE", "COMBINED"]))
        total_ids = len(self.replica_states) or 1
        train_adjust = -selected_len / total_ids

        weight_map = {
            "meta-llama/Llama-2-7b-hf": 0.1,
            "meta-llama/Llama-2-13b-hf": 0.4,
            "meta-llama/Llama-3.1-70B": 0.8,
        }
        model_name = getattr(getattr(replica, "args", None), "model_name", None)
        model_weight = weight_map.get(model_name, 0.1)

        final_score = current_overall + self.alpha * history_adjust + model_weight * train_adjust
        return final_score > self.threshold

    def should_switch_idle_ewma(
        self,
        replica,
        gpu_monitor,
        *,
        time_window: int = 10,
        ewma_alpha: float = 0.3,
        util_lower_bound: float = 25.0,
        quantile_alpha: float = 0.25,
    ):
        if gpu_monitor is None:
            return False

        rid = replica.replica_id
        util_now = gpu_monitor.get_recent_utilization(rid, time_window)
        if util_now is None:
            util_now = 50.0

        if hasattr(replica, "get_queue_length"):
            queue_len_now = replica.get_queue_length()
        elif hasattr(replica, "request_queue"):
            queue_len_now = len(replica.request_queue)
        else:
            queue_len_now = 0

        prev_u = self._ewma_util.get(rid, util_now)
        prev_q = self._ewma_queue.get(rid, float(queue_len_now))

        self._ewma_util[rid] = ewma_alpha * float(util_now) + (1.0 - ewma_alpha) * prev_u
        self._ewma_queue[rid] = ewma_alpha * float(queue_len_now) + (1.0 - ewma_alpha) * prev_q

        if len(self._ewma_util) < 2:
            return False

        all_utils = list(self._ewma_util.values())
        all_queues = list(self._ewma_queue.values())

        u_threshold = min(float(np.quantile(all_utils, quantile_alpha)), float(util_lower_bound))
        q_threshold = float(np.quantile(all_queues, quantile_alpha))

        return self._ewma_util[rid] < u_threshold and self._ewma_queue[rid] < q_threshold

    def should_switch_to_idle(
        self,
        replica,
        replicas,
        state_manager=None,
        gpu_monitor=None,
        process_time: float = 1.0,
        ddl: float = 10.0,
        *,
        strategy: str | None = None,
        **kwargs,
    ):
        use_strategy = strategy or self.idle_strategy
        if use_strategy == "fixed_idle":
            return False
        if use_strategy == "ewma":
            return self.should_switch_idle_ewma(
                replica, gpu_monitor,
                ewma_alpha=kwargs.pop("ewma_alpha", getattr(self, "ewma_alpha", 0.3)),
            )
        return self.should_switch_idle_slo(replica, process_time=process_time, ddl=ddl)


    def check_idle_timeout(self, replica_id):
        if replica_id not in self._idle_since:
            self._idle_since[replica_id] = 0
        self._idle_since[replica_id] += 1
        return self._idle_since[replica_id] >= self.idle_timeout

    def reset_idle_counter(self, replica_id):
        self._idle_since[replica_id] = 0
