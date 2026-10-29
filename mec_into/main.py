"""Entry point and CLI orchestrator for training and evaluation."""

import argparse
import os
import yaml
from mec_into.src.training import train_into, train_ddqn
from mec_into.src.evaluation import ExperimentSuite


def main():
    parser = argparse.ArgumentParser(description="Digital Twin-Driven Intelligent Task Offloading (INTO)")
    parser.add_argument("--config", type=str, default="mec_into/config/experiment_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["all", "train_into", "train_ddqn", "param_analysis", "baseline_compare"], default="all")
    parser.add_argument("--slots", type=int, default=3000, help="Number of time slots for training")
    parser.add_argument("--eval_slots", type=int, default=100, help="Number of slots per evaluation point")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    suite = ExperimentSuite(config)
    checkpoint_into = "outputs/checkpoints/into_model.pt"
    checkpoint_ddqn = "outputs/checkpoints/ddqn_model.pt"

    if args.mode in ["all", "train_into"]:
        print("\n=== Training Proposed Algorithm (INTO) ===")
        into_res = train_into(config, total_slots=args.slots, eval_interval=500, checkpoint_path=checkpoint_into)
        into_agent = into_res["agent"]
    else:
        # Load agent
        from mec_into.src.environment import MECEnvironment
        from mec_into.src.into_agent import INTOAgent
        env = MECEnvironment(config)
        into_agent = INTOAgent(state_dim=env.digital_twin.state_dim, config=config)
        if os.path.exists(checkpoint_into):
            into_agent.load_checkpoint(checkpoint_into)

    if args.mode in ["all", "train_ddqn"]:
        print("\n=== Training Baseline DDQN ===")
        ddqn_res = train_ddqn(config, total_slots=args.slots, eval_interval=500, checkpoint_path=checkpoint_ddqn)
        ddqn_agent = ddqn_res["agent"]
    else:
        from mec_into.src.baselines.ddqn import DDQNBaseline
        from mec_into.src.environment import MECEnvironment
        env = MECEnvironment(config)
        ddqn_agent = DDQNBaseline(state_dim=env.digital_twin.state_dim, config=config)
        if os.path.exists(checkpoint_ddqn):
            ddqn_agent.load_checkpoint(checkpoint_ddqn)

    if args.mode in ["all", "param_analysis"]:
        print("\n=== Parameter Analysis Experiments (Figures 3, 4, 5) ===")
        suite.run_figure_3_learning_rate(total_slots=args.slots)
        suite.run_figure_3_discount_factor(total_slots=args.slots)
        suite.run_figure_4_and_5_parameter_analysis(into_agent, eval_slots=args.eval_slots)

    if args.mode in ["all", "baseline_compare"]:
        print("\n=== Baseline Comparison Experiments (Figure 6) ===")
        # Always reload the best trained INTO and DDQN models for baseline comparison
        if os.path.exists(checkpoint_into):
            print(f"Loading best converged INTO model from {checkpoint_into}")
            into_agent.load_checkpoint(checkpoint_into)
        if os.path.exists(checkpoint_ddqn):
            print(f"Loading best converged DDQN model from {checkpoint_ddqn}")
            ddqn_agent.load_checkpoint(checkpoint_ddqn)
        suite.run_figure_6_baseline_comparisons(into_agent, ddqn_agent, eval_slots=args.eval_slots)

    print("\n[SUCCESS] All requested simulations and visualizations completed!")


if __name__ == "__main__":
    main()
