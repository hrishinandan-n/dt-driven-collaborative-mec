"""Local Priority With Peer Offloading (LPWPO) Baseline.

As defined in Section V-B: Each base station prioritizes the tasks offloaded from
the IoT devices within its coverage. Only when one BS can process all the tasks offloaded
from the IoT devices it serves, can it help other BSs compute tasks.
"""

from typing import List, Tuple
import numpy as np
from ..task_generator import Task


class LPWPOBaseline:
    """LPWPO policy: Local tasks prioritized first; peer offloading only with surplus capacity."""

    def __init__(self, num_bs: int = 3, inter_bs_rate: float = 1.0e8):
        self.num_bs = num_bs
        self.inter_bs_rate = inter_bs_rate

    def select_action(
        self,
        tasks: List[Task],
        energy_buffers: np.ndarray,
        wireless_rates: np.ndarray,
        bs_capacities: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_tasks = len(tasks)
        x = np.zeros(num_tasks, dtype=int)
        y = np.zeros((num_tasks, self.num_bs), dtype=int)
        f = np.zeros((num_tasks, self.num_bs), dtype=float)

        remaining_capacity = [float(cap) for cap in bs_capacities]
        unserved_tasks = []
        task_min_req = {}

        # Phase 1: Local prioritization in each BS
        for bs_id in range(self.num_bs):
            bs_tasks = [m for m, t in enumerate(tasks) if t.bs_id == bs_id]

            candidates = []
            for m in bs_tasks:
                task = tasks[m]
                rate = wireless_rates[m]
                d_w = task.z / rate if rate > 0 else float("inf")
                energy_needed = task.p * d_w

                if energy_needed <= energy_buffers[m] and d_w < task.delta:
                    rem_time = task.delta - d_w
                    if rem_time > 1e-6:
                        l_req = (task.z * task.v) / rem_time
                        priority = task.g / l_req
                        candidates.append((priority, m, l_req))

            # Sort by priority descending
            candidates.sort(key=lambda item: item[0], reverse=True)

            admitted_local = []
            for priority, m, req in candidates:
                if remaining_capacity[bs_id] >= req:
                    admitted_local.append((m, req))
                    remaining_capacity[bs_id] -= req
                    x[m] = 1
                    y[m, bs_id] = 1
                    task_min_req[m] = req
                else:
                    unserved_tasks.append(m)

        # Phase 2: Peer offloading to BSs with remaining surplus capacity
        if unserved_tasks:
            # Sort unserved tasks by award/density
            unserved_tasks.sort(key=lambda m: tasks[m].g / tasks[m].v, reverse=True)

            for m in unserved_tasks:
                task = tasks[m]
                rate = wireless_rates[m]
                d_w = task.z / rate if rate > 0 else float("inf")

                # Try peer BSs
                best_peer = -1
                best_peer_req = float("inf")

                for peer_bs in range(self.num_bs):
                    if peer_bs == task.bs_id:
                        continue
                    if remaining_capacity[peer_bs] <= 0:
                        continue

                    # Wired backhaul delay
                    d_c = task.z / self.inter_bs_rate
                    rem_time = task.delta - d_w - d_c
                    if rem_time > 1e-6:
                        peer_req = (task.z * task.v) / rem_time
                        if peer_req <= remaining_capacity[peer_bs] and peer_req < best_peer_req:
                            best_peer = peer_bs
                            best_peer_req = peer_req

                if best_peer != -1:
                    x[m] = 1
                    y[m, best_peer] = 1
                    remaining_capacity[best_peer] -= best_peer_req
                    task_min_req[m] = best_peer_req

        # Phase 3: Resource allocation of remaining spare capacity
        for target_bs in range(self.num_bs):
            assigned = [m for m in range(num_tasks) if x[m] == 1 and y[m, target_bs] == 1]
            if assigned:
                total_min = sum(task_min_req[m] for m in assigned)
                spare = max(0.0, bs_capacities[target_bs] - total_min)
                for m in assigned:
                    share = (task_min_req[m] / total_min) * spare if total_min > 0 else 0.0
                    f[m, target_bs] = task_min_req[m] + share

        return x, y, f
