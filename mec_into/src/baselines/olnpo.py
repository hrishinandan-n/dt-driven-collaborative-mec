"""Only Local No Peer Offloading (OLNPO) Baseline.

As defined in Section V-B of Zhang et al. (IEEE JSAC 2023):
Each base station can and only can process the tasks which are offloaded from
the IoT devices within its coverage. Peer offloading to other BSs is strictly prohibited.
Because peer offloading is prohibited, base stations with light workloads (e.g. BS-1 with 30 devices)
suffer from low resource utilization, while BSs with heavy workloads (BS-2 with 55 devices and
BS-3 with 80 devices) suffer from severe congestion and task drops.
"""

from typing import List, Tuple
import numpy as np
from ..task_generator import Task


class OLNPOBaseline:
    """OLNPO policy: strictly local task offloading; no peer offloading allowed."""

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

        for bs_id in range(self.num_bs):
            bs_tasks = [m for m, t in enumerate(tasks) if t.bs_id == bs_id]
            capacity = float(bs_capacities[bs_id])
            used_capacity = 0.0
            admitted = []

            # In OLNPO, tasks are submitted by local devices and admitted locally
            for m in bs_tasks:
                task = tasks[m]
                rate = wireless_rates[m]
                if rate <= 0:
                    continue
                d_w = task.z / rate
                energy_needed = task.p * d_w

                # Energy and deadline constraints (Eq. 5)
                if energy_needed <= energy_buffers[m] and d_w < task.delta:
                    rem_time = task.delta - d_w
                    if rem_time > 1e-6:
                        l_req = (task.z * task.v) / rem_time
                        if used_capacity + l_req <= capacity:
                            admitted.append((m, l_req))
                            used_capacity += l_req

            spare = max(0.0, capacity - used_capacity)
            for m, req in admitted:
                x[m] = 1
                y[m, bs_id] = 1
                share = (req / used_capacity) * spare if used_capacity > 0 else 0.0
                f[m, bs_id] = req + share

        return x, y, f
