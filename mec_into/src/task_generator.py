"""Task Generator Module for DT-Driven Collaborative MEC.

Generates computation tasks and IoT device parameters at runtime according to the
stochastic distributions defined in Section V-A of Zhang et al. (IEEE JSAC 2023).
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Task:
    """Represents computation task O_ij(t) = <z_ij(t), v_ij(t), delta_ij(t), g_ij(t)>."""
    bs_id: int              # i in {0, 1, ..., N-1}
    dev_id: int             # j in {0, 1, ..., U_i-1}
    global_id: int          # unique global index in {0, 1, ..., M-1}
    z: float                # Data size z_ij(t) in bits
    v: float                # Processing density v_ij(t) in cycles/bit
    delta: float            # Maximum tolerance delay delta_ij(t) in seconds
    g: float                # Task completion award g_ij(t)
    p: float                # IoT device transmit power p_ij in Watts
    e: float                # Harvested energy arrival e_ij(t) in Joules
    h: float                # Channel power gain h_ij(t)


class TaskGenerator:
    """Samples stochastic tasks and IoT device parameters for each time slot."""

    def __init__(self, config: dict, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.num_bs = config["system"]["num_bs"]
        self.device_distribution = config["system"]["device_distribution"]
        self.total_devices = sum(self.device_distribution)

        # Baseline task parameter bounds
        self.z_range = config["tasks"]["data_size_range"]
        self.v_range = config["tasks"]["processing_density_range"]
        self.delta_range = config["tasks"]["delay_tolerance_range"]
        self.g_range = config["tasks"]["award_range"]
        self.p_range = config["tasks"]["tx_power_range"]
        self.e_range = config["tasks"]["harvested_energy_range"]
        self.nominal_h = config["tasks"]["channel_gain_nominal"]

    def set_seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def generate_tasks_for_slot(
        self,
        t: int,
        award_scale_bs1: float = 1.0,
        density_scale_bs2_3: float = 1.0,
        channel_gain_scale: float = 1.0,
        harvested_energy_scale: float = 1.0,
        task_size_scale: float = 1.0,
        task_delay_scale_psi: float = 1.0,
    ) -> List[Task]:
        """Generates 165 tasks for time slot t across BS-1, BS-2, BS-3.

        Applies parameter scale factors for experimental evaluations (Figs 4, 5, 6).
        """
        tasks = []
        global_idx = 0

        for bs_id, num_devices in enumerate(self.device_distribution):
            for dev_id in range(num_devices):
                # 1. Task data size z_ij(t) ~ U[50, 80] kbit scaled by chi (Fig 6a)
                z = self.rng.uniform(self.z_range[0], self.z_range[1]) * task_size_scale

                # 2. Processing density v_ij(t) ~ U[3000, 8000] cycles/bit
                v = self.rng.uniform(self.v_range[0], self.v_range[1])
                # Scaling for BS-2 and BS-3 only (Fig 4b)
                if bs_id in [1, 2]:
                    v *= density_scale_bs2_3

                # 3. Maximum delay delta_ij(t) ~ U[0.2 * psi, 2.0] s (Fig 6c)
                min_delta = self.delta_range[0] * task_delay_scale_psi
                max_delta = self.delta_range[1]
                if min_delta >= max_delta:
                    min_delta = max_delta * 0.99
                delta = self.rng.uniform(min_delta, max_delta)

                # 4. Task award g_ij(t) ~ U[1, 3]
                g = self.rng.uniform(self.g_range[0], self.g_range[1])
                # Scaling for BS-1 only (Fig 4a)
                if bs_id == 0:
                    g *= award_scale_bs1

                # 5. Device transmit power p_ij ~ U[50, 180] mW
                p = self.rng.uniform(self.p_range[0], self.p_range[1])

                # 6. Harvested energy e_ij(t) ~ U[1, 1.5] mW scaled by rho (Fig 5b)
                e = self.rng.uniform(self.e_range[0], self.e_range[1]) * harvested_energy_scale

                # 7. Channel power gain h_ij(t) scaled by sigma (Fig 5a)
                # Channel gain follows Rayleigh fading with nominal power gain
                fading = self.rng.exponential(scale=1.0)
                h = self.nominal_h * fading * channel_gain_scale

                task = Task(
                    bs_id=bs_id,
                    dev_id=dev_id,
                    global_id=global_idx,
                    z=z,
                    v=v,
                    delta=delta,
                    g=g,
                    p=p,
                    e=e,
                    h=h,
                )
                tasks.append(task)
                global_idx += 1

        return tasks
