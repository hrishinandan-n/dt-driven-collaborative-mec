"""Local Priority With Peer Offloading (LPWPO) Baseline.

As defined in Section V-B of Zhang et al. (IEEE JSAC 2023):
Each base station prioritizes the tasks offloaded from the IoT devices within
its coverage. Only when one BS can process all the tasks offloaded from the IoT
devices it serves, can it help other BSs compute tasks.
LPWPO achieves higher income than OLNPO because surplus capacity is shared,
but remains sub-optimal compared to INTO because each BS strictly prioritizes its
local tasks regardless of whether an overloaded neighbor has tasks with higher awards.
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

            # In LPWPO, each BS admits its local tasks first
            # Sort local tasks by required computing resource ascending (shortest job first)
            local_cands = []
            for m in bs_tasks:
                task = tasks[m]
                rate = wireless_rates[m]
                if rate <= 0:
                    continue
                d_w = task.z / rate
                energy_needed = task.p * d_w

                if energy_needed <= energy_buffers[m] and d_w < task.delta:
                    rem_time = task.delta - d_w
                    if rem_time > 1e-6:
                        l_req = (task.z * task.v) / rem_time
                        local_cands.append((l_req, m))

            local_cands.sort(key=lambda item: item[0])

            for req, m in local_cands:
                if remaining_capacity[bs_id] >= req:
                    remaining_capacity[bs_id] -= req
                    x[m] = 1
                    y[m, bs_id] = 1
                    task_min_req[m] = req
                else:
                    unserved_tasks.append(m)

        # Phase 2: Peer offloading to BSs with remaining surplus capacity
        if unserved_tasks:
            # Sort unserved tasks by data size ascending for efficient backhaul transmission
            unserved_tasks.sort(key=lambda m: tasks[m].z)

            for m in unserved_tasks:
                task = tasks[m]
                rate = wireless_rates[m]
                if rate <= 0:
                    continue
                d_w = task.z / rate

                # Find peer BS with sufficient remaining surplus capacity
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

        # Allocate computing resource frequencies
        for bs_id in range(self.num_bs):
            bs_admitted = [m for m in range(num_tasks) if x[m] == 1 and y[m, bs_id] == 1]
            total_req = sum(task_min_req[m] for m in bs_admitted)
            spare = max(0.0, float(bs_capacities[bs_id]) - total_req)

            for m in bs_admitted:
                share = (task_min_req[m] / total_req) * spare if total_req > 0 else 0.0
                f[m, bs_id] = task_min_req[m] + share

        return x, y, f
