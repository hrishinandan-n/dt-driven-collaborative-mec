"""Evaluation Suite for Reproducing Paper Figures 3, 4, 5, and 6.

Figure 3(a): Impact of Learning Rate (0.0001, 0.001, 0.01)
Figure 3(b): Impact of Discount Factor (0.59, 0.79, 0.99)
Figure 4(a): Impact of Task Award (xi = 1, 1.5, 2, 2.5, 3 for BS-1)
Figure 4(b): Impact of Processing Density (zeta = 1, 1.1, 1.2, 1.3, 1.4 for BS-2 & BS-3)
Figure 5(a): Impact of Wireless Channel Gain (sigma = 0.2 to 1.2, step 0.2)
Figure 5(b): Impact of Harvested Energy Arrival Rate (rho = 0.2 to 1.2, step 0.2)
Figure 6(a): Comparison with Task-size scale factor chi = 1.0 to 1.6
Figure 6(b): Comparison with Computing capacity Fi = 5 to 8 GHz
Figure 6(c): Comparison with Task-delay scale factor psi = 1.0 to 4.0
Figure 6(d): Comparison with Channel bandwidth B = 0.6 to 1.6 MHz
"""

import os
from typing import Dict, List
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .environment import MECEnvironment
from .into_agent import INTOAgent
from .baselines.olnpo import OLNPOBaseline
from .baselines.lpwpo import LPWPOBaseline
from .baselines.ddqn import DDQNBaseline
from .training import train_into, train_ddqn


class ExperimentSuite:
    """Executes paper evaluations and generates publication-grade plots and CSVs."""

    def __init__(self, config: dict, output_dir: str = "outputs"):
        self.config = config
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        self.results_dir = os.path.join(output_dir, "results")
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Style configurations
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["axes.linewidth"] = 0.8

    def run_figure_3_learning_rate(self, total_slots: int = 3000):
        """Figure 3(a): Impact of Learning Rate (0.0001, 0.001, 0.01)."""
        print("\n--- Running Figure 3(a): Impact of Learning Rate ---")
        lrs = [0.0001, 0.001, 0.01]
        data = {}

        for lr in lrs:
            print(f"--> Training INTO with learning rate: {lr}")
            cfg = copy.deepcopy(self.config)
            ckpt_lr_path = os.path.join(self.checkpoints_dir, f"into_lr_{lr}.pt")
            res = train_into(cfg, total_slots=total_slots, lr_override=lr, eval_interval=500, checkpoint_path=ckpt_lr_path)
            data[f"lr_{lr}"] = res["history"]["average_income"]
            data["time_slot"] = res["history"]["slots"]

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.results_dir, "figure_3a_learning_rate.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved numerical data to {csv_path}")

        # Plot Fig 3a
        plt.figure(figsize=(7, 5))
        plt.plot(df["time_slot"], df["lr_0.0001"], "k--", label="Learning rate = 0.0001", linewidth=1.5)
        plt.plot(df["time_slot"], df["lr_0.001"], "b-", label="Learning rate = 0.001", linewidth=1.5)
        plt.plot(df["time_slot"], df["lr_0.01"], "r:", label="Learning rate = 0.01", linewidth=1.5)
        plt.xlabel("Time slot", fontsize=12)
        plt.ylabel("Average income", fontsize=12)
        plt.title("(a) Impact of learning rate", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "figure_3a_learning_rate.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path}")

    def run_figure_3_discount_factor(self, total_slots: int = 3000):
        """Figure 3(b): Impact of Discount Factor (0.59, 0.79, 0.99)."""
        print("\n--- Running Figure 3(b): Impact of Discount Factor ---")
        gammas = [0.99, 0.79, 0.59]
        data = {}

        for gamma in gammas:
            print(f"--> Training INTO with discount factor gamma: {gamma}")
            cfg = copy.deepcopy(self.config)
            ckpt_gamma_path = os.path.join(self.checkpoints_dir, f"into_gamma_{gamma}.pt")
            res = train_into(cfg, total_slots=total_slots, gamma_override=gamma, eval_interval=500, checkpoint_path=ckpt_gamma_path)
            data[f"gamma_{gamma}"] = res["history"]["average_income"]
            data["time_slot"] = res["history"]["slots"]

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.results_dir, "figure_3b_discount_factor.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved numerical data to {csv_path}")

        # Plot Fig 3b
        plt.figure(figsize=(7, 5))
        plt.plot(df["time_slot"], df["gamma_0.99"], "k--", label=r"$\gamma = 0.99$", linewidth=1.5)
        plt.plot(df["time_slot"], df["gamma_0.79"], "b-", label=r"$\gamma = 0.79$", linewidth=1.5)
        plt.plot(df["time_slot"], df["gamma_0.59"], "r:", label=r"$\gamma = 0.59$", linewidth=1.5)
        plt.xlabel("Time slot", fontsize=12)
        plt.ylabel("Average income", fontsize=12)
        plt.title("(b) Impact of discount factor", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "figure_3b_discount_factor.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path}")

    def run_figure_4_and_5_parameter_analysis(self, trained_into_agent: INTOAgent, eval_slots: int = 100):
        """Runs Figures 4(a), 4(b), 5(a), and 5(b) - Offloading device counts per BS."""
        print("\n--- Running Figures 4 & 5: Parameter Analysis ---")
        env = MECEnvironment(self.config, seed=123)

        # 1. Figure 4(a): Impact of Task Award (xi = 1 to 3, step 0.5 for BS-1)
        xi_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        bs1_counts, bs2_counts, bs3_counts = [], [], []

        for xi in xi_values:
            state = env.reset(seed=123, award_scale_bs1=xi)
            tot_bs = np.zeros(3)
            for _ in range(eval_slots):
                action = trained_into_agent.select_action(state, add_noise=False)
                state, _, _, info = env.step(action, award_scale_bs1=xi)
                tot_bs += np.array(info["bs_offloaded_counts"])
            avg_bs = tot_bs / eval_slots
            bs1_counts.append(avg_bs[0])
            bs2_counts.append(avg_bs[1])
            bs3_counts.append(avg_bs[2])

        df_4a = pd.DataFrame({
            "award_scale_factor": xi_values,
            "BS-1": bs1_counts,
            "BS-2": bs2_counts,
            "BS-3": bs3_counts,
        })
        df_4a.to_csv(os.path.join(self.results_dir, "figure_4a_task_award.csv"), index=False)

        # Bar plot Fig 4a
        x = np.arange(len(xi_values))
        width = 0.25
        plt.figure(figsize=(7, 5))
        plt.bar(x - width, bs1_counts, width, label="BS-1", color="#3b82f6")
        plt.bar(x, bs2_counts, width, label="BS-2", color="#f97316")
        plt.bar(x + width, bs3_counts, width, label="BS-3", color="#22c55e")
        plt.xticks(x, [str(v) for v in xi_values])
        plt.xlabel("Award scale factor", fontsize=12)
        plt.ylabel("Number of IoT devices", fontsize=12)
        plt.title("(a) Impact of task award", fontsize=12)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_4a_task_award.png"), dpi=300)
        plt.close()

        # 2. Figure 4(b): Impact of Task Processing Density (zeta = 1.0 to 1.4, step 0.1 for BS-2 & BS-3)
        zeta_values = [1.0, 1.1, 1.2, 1.3, 1.4]
        bs1_b, bs2_b, bs3_b = [], [], []

        for zeta in zeta_values:
            state = env.reset(seed=123, density_scale_bs2_3=zeta)
            tot_bs = np.zeros(3)
            for _ in range(eval_slots):
                action = trained_into_agent.select_action(state, add_noise=False)
                state, _, _, info = env.step(action, density_scale_bs2_3=zeta)
                tot_bs += np.array(info["bs_offloaded_counts"])
            avg_bs = tot_bs / eval_slots
            bs1_b.append(avg_bs[0])
            bs2_b.append(avg_bs[1])
            bs3_b.append(avg_bs[2])

        df_4b = pd.DataFrame({
            "processing_density_scale_factor": zeta_values,
            "BS-1": bs1_b,
            "BS-2": bs2_b,
            "BS-3": bs3_b,
        })
        df_4b.to_csv(os.path.join(self.results_dir, "figure_4b_task_processing_density.csv"), index=False)

        plt.figure(figsize=(7, 5))
        plt.bar(x - width, bs1_b, width, label="BS-1", color="#3b82f6")
        plt.bar(x, bs2_b, width, label="BS-2", color="#f97316")
        plt.bar(x + width, bs3_b, width, label="BS-3", color="#22c55e")
        plt.xticks(x, [str(v) for v in zeta_values])
        plt.xlabel("Processing density scale factor", fontsize=12)
        plt.ylabel("Number of IoT devices", fontsize=12)
        plt.title("(b) Impact of task processing density", fontsize=12)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_4b_task_processing_density.png"), dpi=300)
        plt.close()

        # 3. Figure 5(a): Impact of Wireless Channel Gain (sigma = 0.2 to 1.2, step 0.2)
        sigma_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        bs1_5a, bs2_5a, bs3_5a = [], [], []

        for sigma in sigma_values:
            state = env.reset(seed=123, channel_gain_scale=sigma)
            tot_bs = np.zeros(3)
            for _ in range(eval_slots):
                action = trained_into_agent.select_action(state, add_noise=False)
                state, _, _, info = env.step(action, channel_gain_scale=sigma)
                tot_bs += np.array(info["bs_offloaded_counts"])
            avg_bs = tot_bs / eval_slots
            bs1_5a.append(avg_bs[0])
            bs2_5a.append(avg_bs[1])
            bs3_5a.append(avg_bs[2])

        df_5a = pd.DataFrame({
            "channel_gain_scale_factor": sigma_values,
            "BS-1": bs1_5a,
            "BS-2": bs2_5a,
            "BS-3": bs3_5a,
        })
        df_5a.to_csv(os.path.join(self.results_dir, "figure_5a_wireless_channel_gain.csv"), index=False)

        x5 = np.arange(len(sigma_values))
        plt.figure(figsize=(7, 5))
        plt.bar(x5 - width, bs1_5a, width, label="BS-1", color="#3b82f6")
        plt.bar(x5, bs2_5a, width, label="BS-2", color="#f97316")
        plt.bar(x5 + width, bs3_5a, width, label="BS-3", color="#22c55e")
        plt.xticks(x5, [f"{v:.1f}" for v in sigma_values])
        plt.xlabel("Channel gain scale factor", fontsize=12)
        plt.ylabel("Number of IoT devices", fontsize=12)
        plt.title("(a) Impact of wireless channel gain", fontsize=12)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_5a_wireless_channel_gain.png"), dpi=300)
        plt.close()

        # 4. Figure 5(b): Impact of Harvested Energy Arrival Rate (rho = 0.2 to 1.2, step 0.2)
        rho_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        bs1_5b, bs2_5b, bs3_5b = [], [], []

        for rho in rho_values:
            state = env.reset(seed=123, harvested_energy_scale=rho)
            tot_bs = np.zeros(3)
            for _ in range(eval_slots):
                action = trained_into_agent.select_action(state, add_noise=False)
                state, _, _, info = env.step(action, harvested_energy_scale=rho)
                tot_bs += np.array(info["bs_offloaded_counts"])
            avg_bs = tot_bs / eval_slots
            bs1_5b.append(avg_bs[0])
            bs2_5b.append(avg_bs[1])
            bs3_5b.append(avg_bs[2])

        df_5b = pd.DataFrame({
            "harvested_energy_scale_factor": rho_values,
            "BS-1": bs1_5b,
            "BS-2": bs2_5b,
            "BS-3": bs3_5b,
        })
        df_5b.to_csv(os.path.join(self.results_dir, "figure_5b_harvested_energy.csv"), index=False)

        plt.figure(figsize=(7, 5))
        plt.bar(x5 - width, bs1_5b, width, label="BS-1", color="#3b82f6")
        plt.bar(x5, bs2_5b, width, label="BS-2", color="#f97316")
        plt.bar(x5 + width, bs3_5b, width, label="BS-3", color="#22c55e")
        plt.xticks(x5, [f"{v:.1f}" for v in rho_values])
        plt.xlabel("Harvested energy arrival rate scale factor", fontsize=12)
        plt.ylabel("Number of IoT devices", fontsize=12)
        plt.title("(b) Impact of harvested energy arrival rate", fontsize=12)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_5b_harvested_energy.png"), dpi=300)
        plt.close()
        print("Completed Figures 4 and 5 parameter analysis.")

    def run_figure_6_baseline_comparisons(
        self,
        into_agent: INTOAgent,
        ddqn_agent: DDQNBaseline,
        eval_slots: int = 100,
    ):
        """Runs the four baseline comparisons of Figure 6 (INTO vs OLNPO vs LPWPO vs DDQN)."""
        print("\n--- Running Figure 6: Baseline Comparisons ---")
        olnpo = OLNPOBaseline(num_bs=3)
        lpwpo = LPWPOBaseline(num_bs=3, inter_bs_rate=self.config["system"]["inter_bs_rate"])

        def evaluate_step(env: MECEnvironment, algo_name: str, state: np.ndarray, **kwargs):
            # Extract task generator kwargs vs step kwargs
            task_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ["award_scale_bs1", "density_scale_bs2_3", "channel_gain_scale",
                         "harvested_energy_scale", "task_size_scale", "task_delay_scale_psi"]
            }
            if algo_name == "INTO":
                proto = into_agent.select_action(state, add_noise=False)
                next_s, inc, _, _ = env.step(proto, **kwargs)
                return next_s, inc
            elif algo_name == "DDQN":
                proto = ddqn_agent.select_action(state, evaluate=True)
                next_s, inc, _, _ = env.step(proto, **kwargs)
                return next_s, inc
            elif algo_name == "OLNPO":
                if "bs_capacity" in kwargs and kwargs["bs_capacity"] is not None:
                    env.digital_twin.bs_capacities = np.full(env.num_bs, kwargs["bs_capacity"], dtype=float)
                x, y, f = olnpo.select_action(
                    env.current_tasks,
                    env.digital_twin.energy_buffers,
                    env.current_wireless_rates,
                    env.digital_twin.bs_capacities,
                )
                # Compute income
                inc = 0.0
                for m, task in enumerate(env.current_tasks):
                    if x[m] == 1:
                        rate = env.current_wireless_rates[m]
                        d_w = task.z / rate if rate > 0 else float("inf")
                        d_m = (task.z * task.v) / f[m, task.bs_id] if f[m, task.bs_id] > 0 else float("inf")
                        if d_w + d_m <= task.delta:
                            inc += task.g
                env.digital_twin.update_energy_buffers(env.current_tasks, x, env.current_wireless_rates)
                env.current_step += 1
                env.current_tasks = env.task_generator.generate_tasks_for_slot(env.current_step, **task_kwargs)
                env.current_wireless_rates = env.digital_twin.compute_wireless_rates(env.current_tasks)
                next_s = env.digital_twin.construct_state_vector(env.current_tasks, env.current_wireless_rates)
                return next_s, inc
            elif algo_name == "LPWPO":
                if "bs_capacity" in kwargs and kwargs["bs_capacity"] is not None:
                    env.digital_twin.bs_capacities = np.full(env.num_bs, kwargs["bs_capacity"], dtype=float)
                x, y, f = lpwpo.select_action(
                    env.current_tasks,
                    env.digital_twin.energy_buffers,
                    env.current_wireless_rates,
                    env.digital_twin.bs_capacities,
                )
                inc = 0.0
                for m, task in enumerate(env.current_tasks):
                    if x[m] == 1:
                        target = np.argmax(y[m])
                        rate = env.current_wireless_rates[m]
                        d_w = task.z / rate if rate > 0 else float("inf")
                        d_c = 0.0 if target == task.bs_id else task.z / env.inter_bs_rate
                        d_m = (task.z * task.v) / f[m, target] if f[m, target] > 0 else float("inf")
                        if d_w + d_c + d_m <= task.delta:
                            inc += task.g
                env.digital_twin.update_energy_buffers(env.current_tasks, x, env.current_wireless_rates)
                env.current_step += 1
                env.current_tasks = env.task_generator.generate_tasks_for_slot(env.current_step, **task_kwargs)
                env.current_wireless_rates = env.digital_twin.compute_wireless_rates(env.current_tasks)
                next_s = env.digital_twin.construct_state_vector(env.current_tasks, env.current_wireless_rates)
                return next_s, inc

        def run_eval_for_algo(algo: str, eval_fn, **kwargs) -> float:
            env = MECEnvironment(self.config, seed=999)
            s = env.reset(seed=999, **kwargs)
            tot_inc = 0.0
            for _ in range(eval_slots):
                s, inc = evaluate_step(env, algo, s, **kwargs)
                tot_inc += inc
            return tot_inc / eval_slots

        # --- Fig 6(a): Task-size scale factor chi = 1.0 to 1.6, step 0.1 ---
        print("Evaluating Figure 6(a): Task-size scale factor chi...")
        chi_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        res_6a = {"size_scale_factor": chi_list, "INTO": [], "LPWPO": [], "OLNPO": [], "DDQN": []}

        for chi in chi_list:
            for algo in ["INTO", "LPWPO", "OLNPO", "DDQN"]:
                avg_inc = run_eval_for_algo(algo, evaluate_step, task_size_scale=chi)
                res_6a[algo].append(avg_inc)

        df_6a = pd.DataFrame(res_6a)
        df_6a.to_csv(os.path.join(self.results_dir, "figure_6a_task_size.csv"), index=False)

        # Plot 6a
        plt.figure(figsize=(7, 5))
        plt.plot(chi_list, df_6a["INTO"], "g-o", label="INTO", linewidth=1.5)
        plt.plot(chi_list, df_6a["LPWPO"], "m-^", label="LPWPO", linewidth=1.5)
        plt.plot(chi_list, df_6a["OLNPO"], "b-s", label="OLNPO", linewidth=1.5)
        plt.plot(chi_list, df_6a["DDQN"], "k-d", label="DDQN", linewidth=1.5)
        plt.xlabel("Size scale factor", fontsize=12)
        plt.ylabel("Average income", fontsize=12)
        plt.title("(a) Average income with different task sizes", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_6a_task_size.png"), dpi=300)
        plt.close()

        # --- Fig 6(b): Computing capacity = 5 to 8 GHz ---
        print("Evaluating Figure 6(b): Computing capacity...")
        cap_ghz = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
        res_6b = {"computing_capacity_ghz": cap_ghz, "INTO": [], "LPWPO": [], "OLNPO": [], "DDQN": []}

        for cap in cap_ghz:
            cap_hz = cap * 1e9
            for algo in ["INTO", "LPWPO", "OLNPO", "DDQN"]:
                avg_inc = run_eval_for_algo(algo, evaluate_step, bs_capacity=cap_hz)
                res_6b[algo].append(avg_inc)

        df_6b = pd.DataFrame(res_6b)
        df_6b.to_csv(os.path.join(self.results_dir, "figure_6b_computing_capacity.csv"), index=False)

        plt.figure(figsize=(7, 5))
        plt.plot(cap_ghz, df_6b["INTO"], "g-o", label="INTO", linewidth=1.5)
        plt.plot(cap_ghz, df_6b["LPWPO"], "m-^", label="LPWPO", linewidth=1.5)
        plt.plot(cap_ghz, df_6b["OLNPO"], "b-s", label="OLNPO", linewidth=1.5)
        plt.plot(cap_ghz, df_6b["DDQN"], "k-d", label="DDQN", linewidth=1.5)
        plt.xlabel("Computing capacity of BSs (GHz)", fontsize=12)
        plt.ylabel("Average income", fontsize=12)
        plt.title("(b) Average income with different computing capacities", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_6b_computing_capacity.png"), dpi=300)
        plt.close()

        # --- Fig 6(c): Task-delay scale factor psi = 1.0 to 4.0, step 0.5 ---
        print("Evaluating Figure 6(c): Task-delay scale factor psi...")
        psi_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        res_6c = {"delay_scale_factor": psi_list, "INTO": [], "LPWPO": [], "OLNPO": [], "DDQN": []}

        for psi in psi_list:
            for algo in ["INTO", "LPWPO", "OLNPO", "DDQN"]:
                avg_inc = run_eval_for_algo(algo, evaluate_step, task_delay_scale_psi=psi)
                res_6c[algo].append(avg_inc)

        df_6c = pd.DataFrame(res_6c)
        df_6c.to_csv(os.path.join(self.results_dir, "figure_6c_task_delay.csv"), index=False)

        plt.figure(figsize=(7, 5))
        plt.plot(psi_list, df_6c["INTO"], "g-o", label="INTO", linewidth=1.5)
        plt.plot(psi_list, df_6c["LPWPO"], "m-^", label="LPWPO", linewidth=1.5)
        plt.plot(psi_list, df_6c["OLNPO"], "b-s", label="OLNPO", linewidth=1.5)
        plt.plot(psi_list, df_6c["DDQN"], "k-d", label="DDQN", linewidth=1.5)
        plt.xlabel("Task delay scale factor", fontsize=12)
        plt.ylabel("Average income", fontsize=12)
        plt.title("(c) Average income with different task delays", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_6c_task_delay.png"), dpi=300)
        plt.close()

        # --- Fig 6(d): Channel bandwidth = 0.6 to 1.6 MHz, step 0.2 ---
        print("Evaluating Figure 6(d): Channel bandwidth...")
        bw_list = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        res_6d = {"channel_bandwidth_mhz": bw_list, "INTO": [], "LPWPO": [], "OLNPO": [], "DDQN": []}

        for bw in bw_list:
            # We scale channel bandwidth by altering system bandwidth in config
            cfg_bw = copy.deepcopy(self.config)
            cfg_bw["system"]["bandwidth"] = bw * 1e6
            for algo in ["INTO", "LPWPO", "OLNPO", "DDQN"]:
                env_bw = MECEnvironment(cfg_bw, seed=999)
                s = env_bw.reset(seed=999)
                tot_inc = 0.0
                for _ in range(eval_slots):
                    s, inc = evaluate_step(env_bw, algo, s)
                    tot_inc += inc
                res_6d[algo].append(tot_inc / eval_slots)

        df_6d = pd.DataFrame(res_6d)
        df_6d.to_csv(os.path.join(self.results_dir, "figure_6d_channel_bandwidth.csv"), index=False)

        plt.figure(figsize=(7, 5))
        plt.plot(bw_list, df_6d["INTO"], "g-o", label="INTO", linewidth=1.5)
        plt.plot(bw_list, df_6d["LPWPO"], "m-^", label="LPWPO", linewidth=1.5)
        plt.plot(bw_list, df_6d["OLNPO"], "b-s", label="OLNPO", linewidth=1.5)
        plt.plot(bw_list, df_6d["DDQN"], "k-d", label="DDQN", linewidth=1.5)
        plt.xlabel("Channel bandwidth (MHz)", fontsize=12)
        plt.ylabel("Average income", fontsize=12)
        plt.title("(d) Average income with different channel bandwidths", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "figure_6d_channel_bandwidth.png"), dpi=300)
        plt.close()
        print("Completed Figure 6 baseline comparisons.")
