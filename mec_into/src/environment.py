"""MEC Environment built with EdgeSimPy foundation.

Models physical base stations, edge servers, communication links, task completion delays,
energy consumption, and income reward calculation according to Zhang et al. (IEEE JSAC 2023).
"""

from typing import List, Tuple
import numpy as np
import edge_sim_py

from .task_generator import Task, TaskGenerator
from .digital_twin import DigitalTwin
from .action_refinement import FastActionRefiner


class MECEnvironment:
    """Collaborative Mobile Edge Computing Environment backed by EdgeSimPy components."""

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.num_bs = int(config["system"]["num_bs"])
        self.device_distribution = [int(x) for x in config["system"]["device_distribution"]]
        self.total_devices = sum(self.device_distribution)
        self.inter_bs_rate = float(config["system"]["inter_bs_rate"])
        self.base_capacity = float(config["system"]["bs_computing_capacity"])

        # Initialize Task Generator and Digital Twin
        self.task_generator = TaskGenerator(config, seed=seed)
        self.digital_twin = DigitalTwin(config)
        self.refiner = FastActionRefiner(num_bs=self.num_bs)

        # Build underlying EdgeSimPy simulator components
        self._init_edgesimpy_topology()

        self.current_step = 0
        self.current_tasks: List[Task] = []
        self.current_wireless_rates: np.ndarray = np.zeros(self.total_devices)

    def _init_edgesimpy_topology(self):
        """Initializes EdgeSimPy BaseStation, EdgeServer, and NetworkLink components."""
        self.sim = edge_sim_py.Simulator(tick_duration=1, tick_unit="seconds")

        self.bs_nodes = []
        self.edge_servers = []

        # Create 3 Base Stations and Edge Servers
        for i in range(self.num_bs):
            bs = edge_sim_py.BaseStation()
            bs.coordinates = (i * 500, 0)
            self.bs_nodes.append(bs)

            # Edge server with 5 GHz CPU capacity (represented in MHz)
            server = edge_sim_py.EdgeServer()
            server.cpu = int(self.base_capacity / 1e6)
            server.base_station = bs
            self.edge_servers.append(server)

        # Create optical fiber links between BSs (100 Mbps)
        self.network_links = []
        for i in range(self.num_bs):
            for j in range(i + 1, self.num_bs):
                link = edge_sim_py.NetworkLink()
                link.nodes = [self.bs_nodes[i], self.bs_nodes[j]]
                link.bandwidth = self.inter_bs_rate / 1e6 # Mbps
                link.delay = 0.001
                self.network_links.append(link)

    def reset(
        self,
        seed: int = None,
        bs_capacity: float = None,
        award_scale_bs1: float = 1.0,
        density_scale_bs2_3: float = 1.0,
        channel_gain_scale: float = 1.0,
        harvested_energy_scale: float = 1.0,
        task_size_scale: float = 1.0,
        task_delay_scale_psi: float = 1.0,
    ) -> np.ndarray:
        """Resets the environment for a new episode."""
        if seed is not None:
            self.seed = seed
            self.task_generator.set_seed(seed)

        self.current_step = 0

        # Initialize device energy buffers uniformly in [0.01, 0.05] Joules
        init_energy_range = self.config["tasks"]["initial_battery_range"]
        rng = np.random.RandomState(self.seed)
        initial_energy = rng.uniform(init_energy_range[0], init_energy_range[1], size=self.total_devices)

        # Reset Digital Twin
        self.digital_twin.reset(initial_energy, bs_capacity=bs_capacity)

        # Generate initial workload for slot 0
        self.current_tasks = self.task_generator.generate_tasks_for_slot(
            t=0,
            award_scale_bs1=award_scale_bs1,
            density_scale_bs2_3=density_scale_bs2_3,
            channel_gain_scale=channel_gain_scale,
            harvested_energy_scale=harvested_energy_scale,
            task_size_scale=task_size_scale,
            task_delay_scale_psi=task_delay_scale_psi,
        )

        # Compute wireless rates in Digital Twin
        self.current_wireless_rates = self.digital_twin.compute_wireless_rates(self.current_tasks)

        # Return initial state vector
        return self.digital_twin.construct_state_vector(self.current_tasks, self.current_wireless_rates)

    def step(
        self,
        raw_action: np.ndarray,
        bs_capacity: float = None,
        award_scale_bs1: float = 1.0,
        density_scale_bs2_3: float = 1.0,
        channel_gain_scale: float = 1.0,
        harvested_energy_scale: float = 1.0,
        task_size_scale: float = 1.0,
        task_delay_scale_psi: float = 1.0,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """Performs one time slot transition.

        1. Refines proto-action using Algorithm 1.
        2. Executes task offloading, transmission, and computation.
        3. Computes system income r(t) according to Eq. (11).
        4. Evolves device battery levels according to Eq. (6).
        5. Generates tasks for next slot and returns next state s(t+1).
        """
        if bs_capacity is not None:
            self.digital_twin.bs_capacities = np.full(self.num_bs, bs_capacity, dtype=float)
        # Refine action using Algorithm 1
        x, y, f, ref_stats = self.refiner.refine_action(
            continuous_action=raw_action,
            tasks=self.current_tasks,
            bs_computing_capacities=self.digital_twin.bs_capacities,
            energy_buffers=self.digital_twin.energy_buffers,
            wireless_rates=self.current_wireless_rates,
            inter_bs_rate=self.inter_bs_rate,
        )

        # Compute delays and verify deadlines for reward calculation (Eq. 9, 8, 4, 11)
        completed_tasks_count = 0
        total_income = 0.0
        bs_offloaded_counts = [0, 0, 0]

        for m, task in enumerate(self.current_tasks):
            if x[m] == 1:
                bs_offloaded_counts[task.bs_id] += 1
                rate = self.current_wireless_rates[m]
                d_w = task.z / rate if rate > 0 else float("inf")

                # Find which BS processed this task
                target_bs = np.argmax(y[m])
                if target_bs == task.bs_id:
                    d_c = 0.0
                else:
                    d_c = task.z / self.inter_bs_rate

                f_alloc = f[m, target_bs]
                if f_alloc > 0:
                    d_m = (task.z * task.v) / f_alloc
                else:
                    d_m = float("inf")

                # Total completion duration Eq. (11): d_ij(t) = d_w + d_c + d_m
                d_total = d_w + d_c + d_m

                # Reward income: sum_{i,j} g_ij(t) * x_ij(t) * 1{d_ij(t) <= delta_ij(t)}
                if d_total <= task.delta:
                    total_income += task.g
                    completed_tasks_count += 1

        # Battery evolution according to Eq. (6)
        self.digital_twin.update_energy_buffers(self.current_tasks, x, self.current_wireless_rates)

        self.current_step += 1

        # Sample new tasks for next time slot
        self.current_tasks = self.task_generator.generate_tasks_for_slot(
            t=self.current_step,
            award_scale_bs1=award_scale_bs1,
            density_scale_bs2_3=density_scale_bs2_3,
            channel_gain_scale=channel_gain_scale,
            harvested_energy_scale=harvested_energy_scale,
            task_size_scale=task_size_scale,
            task_delay_scale_psi=task_delay_scale_psi,
        )

        # Recalculate wireless transmission rates
        self.current_wireless_rates = self.digital_twin.compute_wireless_rates(self.current_tasks)

        # Construct next state s(t+1)
        next_state = self.digital_twin.construct_state_vector(self.current_tasks, self.current_wireless_rates)

        info = {
            "income": total_income,
            "completed_tasks": completed_tasks_count,
            "bs_offloaded_counts": bs_offloaded_counts,
            "refinement_stats": ref_stats,
            "x": x,
            "y": y,
            "f": f,
        }

        done = False
        return next_state, total_income, done, info
