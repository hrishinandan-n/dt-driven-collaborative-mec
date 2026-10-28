"""Only Local No Peer Offloading (OLNPO) Baseline.

As defined in Section V-B: Each base station can and only can process the tasks
which are offloaded from the IoT devices within its coverage. Peer offloading
to other BSs is strictly prohibited.
"""

from typing import List
import numpy as np
from ..task_generator import Task


class OLNPOBaseline:
    """OLNPO policy: No peer offloading allowed between base stations."""

    def __init__(self, num_bs: int = 3):
        self.num_bs = num_bs

    def select_action(
        self,
        tasks: List[Task],
        energy_buffers: np.ndarray,
        wireless_rates: np.ndarray,
        bs_capacities: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decides task offloading strictly within each BS coverage area."""
        num_tasks = len(tasks)
        x = np.zeros(num_tasks, dtype=int)
        y = np.zeros((num_tasks, self.num_bs), dtype=int)
        f = np.zeros((num_tasks, self.num_bs), dtype=float)

        # Process BS by BS independently
        for bs_id in range(self.num_bs):
            bs_tasks = [m for m, t in enumerate(tasks) if t.bs_id == bs_id]

            # Candidate tasks that satisfy energy constraint (5)
            candidates = []
            min_cpu_req = {}

            for m in bs_tasks:
                task = tasks[m]
                rate = wireless_rates[m]
                d_w = task.z / rate if rate > 0 else float("inf")
                energy_needed = task.p * d_w

                if energy_needed <= energy_buffers[m] and d_w < task.delta:
                    # Minimum CPU requirement l_i_ij(t) for local execution (d_c = 0)
                    rem_time = task.delta - d_w
                    if rem_time > 1e-6:
                        l_req = (task.z * task.v) / rem_time
                        if l_req <= bs_capacities[bs_id]:
                            priority = task.g / l_req
                            candidates.append((priority, m))
                            min_cpu_req[m] = l_req

            # Prioritize tasks with higher g / l_req
            candidates.sort(key=lambda item: item[0], reverse=True)

            # Admit tasks up to local BS capacity
            capacity = bs_capacities[bs_id]
            used_capacity = 0.0
            admitted = []

            for priority, m in candidates:
                req = min_cpu_req[m]
                if used_capacity + req <= capacity:
                    admitted.append(m)
                    used_capacity += req

            # Allocate CPU
            spare = max(0.0, capacity - used_capacity)
            for m in admitted:
                x[m] = 1
                y[m, bs_id] = 1
                share = (min_cpu_req[m] / used_capacity) * spare if used_capacity > 0 else 0.0
                f[m, bs_id] = min_cpu_req[m] + share

        return x, y, f
