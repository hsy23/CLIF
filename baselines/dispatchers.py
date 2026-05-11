
import threading
import time
import uuid
import math
import heapq
from collections import deque
from core.dispatcher import BaseDispatcher



class RoundDispatcher(BaseDispatcher):
    def __init__(self, replicas, ddl):
        super().__init__(replicas, ddl)
        self.request_ddl = ddl
        self.current_index = 0
        self.dispatch_lock = threading.Lock()
        self.dispatch_thread = None
        self.queue_times = []
        self.queue_time_window = 50
        self.avg_queue_time = 0.0
        self.min_infer_batch_size = min((r.infer_batch_size for r in replicas), default=1)
        self.replica_by_id = {r.replica_id: r for r in replicas}
        self.replica_batch_config = {
            r.replica_id: {
                "batch_size": r.infer_batch_size,
                "max_batch_size": r.infer_batch_size,
                "train_batch_size": r.train_batch_size,
            }
            for r in replicas
        }

    def start(self):
        if self.dispatch_thread is not None and self.dispatch_thread.is_alive():
            return
        self.running = True
        self.dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatch_thread.start()
        print("[RoundDispatcher] Started")

    def _dispatch_loop(self):
        batch_size = 3
        while self.running:
            requests = []
            with self.queue_lock:
                i = 0
                while self.request_queue and i < batch_size:
                    req = self.request_queue.popleft()
                    if req["timestamp"] + self.ddl - 0.1 >= time.time():
                        requests.append(req)
                        i += 1

            if not requests:
                time.sleep(0.1)
                continue

            with self.dispatch_lock:
                replica = self.replicas[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.replicas)

            for req in requests:
                success = replica.add_request(req)
                if success:
                    now = time.time()
                    queue_time = now - req["timestamp"]
                    self.metrics.append({
                        "request_id": req["request_id"],
                        "replica_id": replica.replica_id,
                        "timestamp": now,
                        "queue_time": queue_time,
                    })
                    self._update_avg_queue_time(queue_time)

    def stop(self):
        self.running = False
        if self.dispatch_thread and self.dispatch_thread.is_alive():
            self.dispatch_thread.join()
            print("[RoundDispatcher] Stopped")


    def estimate_recent_pre_service_wait(self, replica_ids=None, window=50):
        samples = []
        replica_id_set = set(replica_ids) if replica_ids is not None else None
        for replica in self.replicas:
            if replica_id_set is not None and replica.replica_id not in replica_id_set:
                continue
            serve_metrics = replica.get_metrics().get("serve_metrics", [])
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
        replica_id_set = set(replica_ids) if replica_ids is not None else None
        samples = []
        for replica in self.replicas:
            if replica_id_set is not None and replica.replica_id not in replica_id_set:
                continue
            for metric in replica.get_metrics().get("serve_metrics", [])[-20:]:
                process_time = metric.get("process_time")
                infer_batch_size = metric.get("infer_batch_size")
                if not process_time or not infer_batch_size:
                    continue
                samples.append(max(0.01, process_time / infer_batch_size * self.min_infer_batch_size))
        if not samples:
            return max(0.05, min(self.request_ddl, 0.1 * self.request_ddl))
        return max(0.05, min(self.request_ddl, sum(samples) / len(samples)))

    def get_available_service_time(self, replica_ids=None):
        pre_service_wait = self.estimate_recent_pre_service_wait(replica_ids)
        min_service_budget = self.get_min_service_time_budget(replica_ids)
        return max(min_service_budget, self.request_ddl - pre_service_wait)

    def update_subflow_config(self, replica_id, batch_size=None, max_batch_size=None,
                              train_batch_size=None, mean_interval=None,
                              std_interval=None, para=None):
        config = self.replica_batch_config.get(replica_id)
        if config is None:
            return False
        replica = self.replica_by_id.get(replica_id)
        if batch_size is not None:
            config["batch_size"] = batch_size
            if replica is not None:
                replica.set_infer_batch_size(batch_size)
        if max_batch_size is not None:
            config["max_batch_size"] = max_batch_size
        if train_batch_size is not None:
            config["train_batch_size"] = train_batch_size
            if replica is not None:
                replica.set_train_batch_size(train_batch_size)
        return True



class dLoRADispatcher(BaseDispatcher):
    def __init__(self, replicas, ddl):
        super().__init__(replicas, ddl)
        self.dispatch_lock = threading.Lock()
        self.dispatch_thread = None
        self.replica_exec_cost = {i: 0 for i in range(len(replicas))}

        self.migration_thread = None
        self.migration_interval = 10
        self.migration_req_thres = 5

    def start(self):
        if self.dispatch_thread is not None and self.dispatch_thread.is_alive():
            return
        self.running = True
        self.dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatch_thread.start()
        print("[dLoRADispatcher] Started")

        self.migration_thread = threading.Thread(target=self._migration_loop, daemon=True)
        self.migration_thread.start()


    def _dispatch_loop(self):
        batch_size = 3
        while self.running:
            requests = []
            with self.queue_lock:
                for _ in range(batch_size):
                    if self.request_queue:
                        requests.append(self.request_queue.popleft())
                    else:
                        break

            if not requests:
                time.sleep(0.1)
                continue

            dispatched = False
            with self.dispatch_lock:
                rid = self._select_optimal_replica()
                if rid is not None:
                    replica = self.replicas[rid]
                    for req in requests:
                        success = replica.add_request(req)
                        if success:
                            dispatched = True
                        self.metrics.append({
                            "request_id": req["request_id"],
                            "replica_id": replica.replica_id,
                            "timestamp": time.time(),
                            "queue_time": time.time() - req["timestamp"],
                        })

            if not dispatched:
                with self.queue_lock:
                    for req in requests:
                        self.request_queue.append(req)
                time.sleep(0.5)

    def _select_optimal_replica(self):
        self._update_replica_exec_cost()
        min_cost = math.inf
        best = None
        for rid in range(len(self.replicas)):
            cost = self.replica_exec_cost[rid]
            if cost < min_cost:
                min_cost = cost
                best = rid
        return best

    def _update_replica_exec_cost(self):
        for rid in range(len(self.replicas)):
            self.replica_exec_cost[rid] = self.replicas[rid].get_queue_length()


    def _migration_loop(self):
        while self.running:
            self._migration_schedule()
            time.sleep(self.migration_interval)

    def _migration_schedule(self):
        replica_req_cnt = {}
        num_reqs = 0
        for rid in range(len(self.replicas)):
            replica = self.replicas[rid]
            replica.request_queue = deque(
                r for r in replica.request_queue
                if r["timestamp"] + self.ddl - 0.1 >= time.time()
            )
            ql = replica.get_queue_length()
            replica_req_cnt[rid] = ql
            num_reqs += ql

        if len(replica_req_cnt) < 2 or num_reqs == 0:
            return

        sorted_replicas = sorted(replica_req_cnt.items(), key=lambda x: x[1])
        matched = []
        while sorted_replicas:
            avg = num_reqs / len(sorted_replicas)
            most_id, most_cnt = sorted_replicas.pop()
            num_reqs -= most_cnt
            delta = most_cnt - avg
            if delta < self.migration_req_thres:
                return

            for rid, cnt in sorted_replicas:
                if delta <= 0 or matched:
                    break
                to_fill = min(avg - cnt, delta)
                if to_fill <= 0:
                    break
                matched.append((rid, int(to_fill)))
                delta -= to_fill

            if matched:
                for target_id, to_migrate in matched:
                    print(f"[dLoRA] Migrating {to_migrate} requests: Replica {most_id} -> Replica {target_id}")
                    self._migrate_requests(most_id, target_id, to_migrate)
                    break

    def _migrate_requests(self, from_id, to_id, count):
        from_replica = self.replicas[from_id]
        to_replica = self.replicas[to_id]
        migrated = from_replica.get_requests_for_migration(count)
        if not migrated:
            return
        ok = 0
        for req in migrated:
            if to_replica.add_request(req):
                ok += 1
            else:
                from_replica.add_request(req)
        print(f"[dLoRA] Migration success={ok}: Replica {from_id} -> Replica {to_id}")
        self.metrics.append({
            "migration_id": str(uuid.uuid4()),
            "from_replica_id": from_id,
            "to_replica_id": to_id,
            "requested_count": count,
            "migrated_count": ok,
            "timestamp": time.time(),
        })

    def stop(self):
        self.running = False
        if self.dispatch_thread and self.dispatch_thread.is_alive():
            self.dispatch_thread.join()
        if self.migration_thread and self.migration_thread.is_alive():
            self.migration_thread.join()
        print("[dLoRADispatcher] Migration thread stopped")



class ShepherdDispatcher(BaseDispatcher):
    def __init__(self, replicas, ddl):
        super().__init__(replicas, ddl)
        self.request_queue = []

        self.model_params = {}
        for replica in self.replicas:
            self.model_params[replica.replica_id] = (0.1, 0.05)

        self.batch_completion_thread = None
        self.model_fitting_thread = None
        self.large_adjustment_interval = 60

        self.queue_times = []
        self.queue_time_window = 100
        self.avg_queue_time = 0


    def add_request(self, request):
        if "request_id" not in request:
            request["request_id"] = str(uuid.uuid4())
        request["timestamp"] = time.time()
        request["deadline"] = request["timestamp"] + request["ddl"]
        with self.queue_lock:
            heapq.heappush(self.request_queue, (request["deadline"], request))
        return request["request_id"]


    def start(self):
        if self.running:
            return
        self.running = True

        self.batch_completion_thread = threading.Thread(
            target=self._handle_batch_completion, daemon=True)
        self.batch_completion_thread.start()

        self.model_fitting_thread = threading.Thread(
            target=self._model_fitting_worker, daemon=True)
        self.model_fitting_thread.start()

        print("[ShepherdDispatcher] Started")

    def stop(self):
        self.running = False
        if self.batch_completion_thread and self.batch_completion_thread.is_alive():
            self.batch_completion_thread.join(timeout=1)
        if self.model_fitting_thread and self.model_fitting_thread.is_alive():
            self.model_fitting_thread.join(timeout=1)
        print("[ShepherdDispatcher] Stopped")


    def _handle_batch_completion(self):
        while self.running:
            rid = 0
            while rid < len(self.replicas):
                if len(self.request_queue) == 0:
                    time.sleep(0.01)
                    continue
                replica = self.replicas[rid]
                if replica.is_idle:
                    self._generate_and_execute_batch(replica)
                rid += 1
            time.sleep(0.01)

    def _generate_and_execute_batch(self, replica):
        batch = self._generate_candidate_batch(replica)
        if batch:
            self._execute_batch(replica, batch)

    def _generate_candidate_batch(self, replica):
        rid = replica.replica_id
        a, b = self.model_params.get(rid, (0.1, 0.05))
        max_bs = self._calculate_max_batch_size(rid)
        if max_bs <= 0:
            return None

        with self.queue_lock:
            if not self.request_queue:
                return None
            queue_copy = list(self.request_queue)

        now = time.time()
        valid = []
        for deadline, req in sorted(queue_copy):
            if now + 0.05 <= deadline and len(valid) < max_bs:
                valid.append((deadline, req))

        if not valid:
            return None

        with self.queue_lock:
            requests = []
            for item in valid:
                if item in self.request_queue:
                    self.request_queue.remove(item)
                    requests.append(item[1])

        if not requests:
            return None
        return {"batch_size": len(requests), "requests": requests}

    def _execute_batch(self, replica, batch):
        rid = replica.replica_id
        replica.set_infer_batch_size(batch["batch_size"])
        for req in batch["requests"]:
            if replica.add_request(req):
                qt = time.time() - req["timestamp"]
                self._update_avg_queue_time(qt)
                self.metrics.append({
                    "request_id": req["request_id"],
                    "replica_id": rid,
                    "timestamp": time.time(),
                    "queue_time": qt,
                    "batch_size": batch["batch_size"],
                })

    def _calculate_max_batch_size(self, replica_id):
        a, b = self.model_params[replica_id]
        earliest = float("inf")
        with self.queue_lock:
            if self.request_queue:
                earliest = self.request_queue[0][0]
        if earliest == float("inf"):
            return 6
        remaining = earliest - time.time()
        if remaining <= 0:
            return 3
        return max(3, int((remaining - b) / a))


    def _model_fitting_worker(self):
        time.sleep(20)
        while self.running:
            try:
                print("[Shepherd] Fitting latency model")
                all_metrics = []
                for ml in self._collect_inference_metrics().values():
                    all_metrics.extend(ml)
                a, b, r2 = self._fit_inference_model(all_metrics, default_a=0.1, default_b=0.05)
                if r2 is not None:
                    print(f"[Shepherd] Fitted model: y = {a:.4f}*x + {b:.4f}, R2={r2:.4f}")
                for replica in self.replicas:
                    self.model_params[replica.replica_id] = (a, b)
                time.sleep(self.large_adjustment_interval)
            except Exception as e:
                print(f"[Shepherd] Model fitting failed: {e}")
                time.sleep(10)


    def update_model_params(self, replica_id, a, b):
        self.model_params[replica_id] = (a, b)

    def get_avg_queue_time(self):
        return self.avg_queue_time
