"""Digital Twin (DT) implementation for Collaborative MEC.

Maintains a high-fidelity virtual mirror of physical entities (IoT devices, BSs, and wireless channels).
Tracks state vectors, supports state observation for DRL policies, and synchronizes with the MEC environment.
"""

from typing import List
import numpy as np
from .task_generator import Task


class DigitalTwin:
    """Virtual representation and state mirror of the collaborative MEC system."""

    def __init__(self, config: dict):
        self.config = config
        self.num_bs = int(config["system"]["num_bs"])
        self.device_distribution = [int(x) for x in config["system"]["device_distribution"]]
        self.total_devices = sum(self.device_distribution)
        self.bandwidth = float(config["system"]["bandwidth"])
        self.noise_psd = float(config["system"]["noise_psd"])
        self.base_capacity = float(config["system"]["bs_computing_capacity"])
        self.inter_bs_rate = float(config["system"]["inter_bs_rate"])

        # Device Energy Buffers Q_ij^E(t)
        self.energy_buffers = np.zeros(self.total_devices, dtype=float)

        # BS Computing capacities F_i(t)
        self.bs_capacities = np.full(self.num_bs, self.base_capacity, dtype=float)

        # Wired transmission rates between BSs C_i'_i(t) (3x3 matrix)
        self.wired_rates = np.full((self.num_bs, self.num_bs), self.inter_bs_rate, dtype=float)
        np.fill_diagonal(self.wired_rates, float("inf"))

        # Cached state dimensions
        # State vector components (Section III-B2):
        # 1. z_ij(t): total_devices (165)
        # 2. v_ij(t): total_devices (165)
        # 3. delta_ij(t): total_devices (165)
        # 4. g_ij(t): total_devices (165)
        # 5. iota_ij(t): total_devices (165)
        # 6. C_i'_i(t): num_bs * (num_bs - 1) = 6 wired links
        # 7. Q_ij^E(t): total_devices (165)
        # 8. F_i(t): num_bs = 3
        # Total dimension = 165*6 + 6 + 3 = 999.
        self.state_dim = self.total_devices * 6 + self.num_bs * (self.num_bs - 1) + self.num_bs

    def reset(self, initial_energy: np.ndarray, bs_capacity: float = None):
        """Resets the Digital Twin at the start of an episode."""
        self.energy_buffers = initial_energy.copy()
        if bs_capacity is not None:
            self.bs_capacities = np.full(self.num_bs, bs_capacity, dtype=float)
        else:
            self.bs_capacities = np.full(self.num_bs, self.base_capacity, dtype=float)

    def compute_wireless_rates(self, tasks: List[Task], x_decisions: np.ndarray = None) -> np.ndarray:
        """Calculates wireless transmission rate iota_ij(t) according to Eq. (2) and Eq. (3).

        Eq. (2): iota_ij(t) = B * log2(1 + (p_ij * h_ij(t)) / (B * N0 + G_ij(t)))
        Eq. (3): G_ij(t) = sum_{{k in U_i \\ {j}}} x_ik(t) * p_ik * h_ik(t)
        When calculating before action (or potential transmission), estimate with prospective transmission.
        """
        rates = np.zeros(self.total_devices, dtype=float)

        # Group device indices by BS
        bs_groups = {i: [] for i in range(self.num_bs)}
        for m, task in enumerate(tasks):
            bs_groups[task.bs_id].append(m)

        for bs_id, dev_indices in bs_groups.items():
            # Signal powers for all devices in this BS
            sig_powers = np.array([tasks[m].p * tasks[m].h for m in dev_indices])
            if x_decisions is not None:
                transmitting = np.array([x_decisions[m] for m in dev_indices])
            else:
                # Digital Twin optimistic channel rate estimation for scheduling
                transmitting = np.zeros(len(dev_indices))

            # Total received interference power in BS with pseudo-orthogonal spreading / quasi-orthogonal channels
            # Spreading factor / cross-correlation factor beta = 0.01
            beta_interf = 0.01
            total_rx_power = np.sum(sig_powers * transmitting) * beta_interf

            for idx_local, m in enumerate(dev_indices):
                p_sig = sig_powers[idx_local]
                # Intra-cell interference: Eq. (3)
                g_ij = max(0.0, total_rx_power - (p_sig * beta_interf if transmitting[idx_local] else 0.0))
                # Channel noise
                noise = self.bandwidth * self.noise_psd
                sinr = p_sig / (noise + g_ij + 1e-15)
                # Rate Eq. (2)
                rate = self.bandwidth * np.log2(1.0 + sinr)
                rates[m] = rate

        return rates

    def construct_state_vector(self, tasks: List[Task], wireless_rates: np.ndarray) -> np.ndarray:
        """Constructs the exact MDP state vector s(t) defined in Section III-B2.

        Normalizes values to [0, 1] range for stable neural network training.
        """
        z_vec = np.array([task.z for task in tasks]) / 100000.0        # Normalized by 100 kbit
        v_vec = np.array([task.v for task in tasks]) / 10000.0         # Normalized by 10000 cycles/bit
        delta_vec = np.array([task.delta for task in tasks]) / 2.0     # Normalized by 2.0 s
        g_vec = np.array([task.g for task in tasks]) / 3.0             # Normalized by 3.0
        iota_vec = wireless_rates / 2000000.0                         # Normalized by 2 Mbps
        q_vec = self.energy_buffers / 0.1                              # Normalized by 0.1 J

        # Wired rates between distinct BS pairs
        c_wired = []
        for i in range(self.num_bs):
            for i_prime in range(self.num_bs):
                if i != i_prime:
                    c_wired.append(self.wired_rates[i, i_prime] / 100000000.0)
        c_wired_vec = np.array(c_wired)

        # BS capacities
        f_vec = self.bs_capacities / 10.0e9

        state = np.concatenate([
            z_vec,
            v_vec,
            delta_vec,
            g_vec,
            iota_vec,
            c_wired_vec,
            q_vec,
            f_vec,
        ]).astype(np.float32)

        return state

    def update_energy_buffers(self, tasks: List[Task], x: np.ndarray, wireless_rates: np.ndarray):
        """Updates device battery energy buffer Q_ij^E(t+1) according to Eq. (6).

        Q_ij^E(t+1) = Q_ij^E(t) + e_ij(t) - x_ij(t) * p_ij * d_ij^w(t)
        """
        for m, task in enumerate(tasks):
            if x[m] == 1:
                rate = wireless_rates[m]
                d_w = task.z / rate if rate > 0 else 0.0
                tx_energy = task.p * d_w
            else:
                tx_energy = 0.0

            # Battery evolution
            self.energy_buffers[m] = max(0.0, self.energy_buffers[m] + task.e - tx_energy)
