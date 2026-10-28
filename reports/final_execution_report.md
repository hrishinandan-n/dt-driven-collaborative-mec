# Final Technical Execution Report: Digital Twin-Driven Intelligent Task Offloading (INTO)

**Paper Reference:** *Digital Twin-Driven Intelligent Task Offloading for Collaborative Mobile Edge Computing*  
**Publication:** IEEE Journal on Selected Areas in Communications (JSAC), Vol. 41, No. 10, October 2023  
**Authors:** Yongchao Zhang, Jia Hu, and Geyong Min  

---

## 1. Executive Summary

This project implements a complete, runnable, and rigorous simulation of the **INTO (Intelligent Task Offloading)** algorithm and collaborative Mobile Edge Computing (MEC) framework described in Zhang et al. (IEEE JSAC 2023).

The implementation incorporates:
1. **Digital Twin (DT):** A virtual mirror of physical IoT devices, cellular base stations, inter-BS optical communication links, battery energy buffers, and wireless channel states.
2. **Fast Action Refinement (Algorithm 1):** Maps continuous actor outputs into feasible discrete offloading choices adhering to energy budgets, strict latency deadlines, and MEC computing capacity constraints.
3. **DRL-Based INTO Scheme (Algorithm 2):** Deep Deterministic Policy Gradient (DDPG) with target actor/critic networks and replay memory optimizing peer offloading and computing resource allocation.
4. **Three Paper Baselines:** 
   - **OLNPO** (*Only Local No Peer Offloading*): Restricts task computation strictly to serving base stations.
   - **LPWPO** (*Local Priority With Peer Offloading*): Prioritizes local tasks and utilizes only leftover capacity for peer offloading.
   - **DDQN** (*Double Deep Q-Network*): Value-based DRL benchmark.
5. **Full Experimental Reproduction:**
   - Parameter analysis matching Figures 3(a), 3(b), 4(a), 4(b), 5(a), and 5(b).
   - Baseline comparisons matching Figures 6(a), 6(b), 6(c), and 6(d).

---

## 2. Equation-by-Equation Reference & Verification

Every equation from the paper has been verified and implemented directly in the codebase:

| Eq. | Purpose | Mathematical Expression | Key Variables & Units | Source Code Mapping |
| :---: | :--- | :--- | :--- | :--- |
| **(1)** | Device DT Digital Model | $\mathcal{D}^K_{ij} = \{p_{ij}, Q^E_{ij}(t), e_{ij}(t), O_{ij}(t)\}$ | $p_{ij}\text{ (W)}, Q^E_{ij}\text{ (J)}, e_{ij}\text{ (J)}$ | [`digital_twin.py::DigitalTwin`](mec_into/src/digital_twin.py) |
| **(2)** | Wireless Transmission Rate | $\iota_{ij}(t) = B \log_2\left(1 + \frac{p_{ij} h_{ij}(t)}{B N_0 + G_{ij}(t)}\right)$ | $B = 1\text{ MHz}, N_0 = 10^{-8}\text{ W/Hz}, \iota\text{ (bps)}$ | [`digital_twin.py::compute_wireless_rates`](mec_into/src/digital_twin.py) |
| **(3)** | Intra-cell Interference | $G_{ij}(t) = \sum_{k \in \mathcal{U}_i \setminus \{j\}} x_{ik}(t) p_{ik} h_{ik}(t)$ | Power sum over co-channel transmitting devices | [`digital_twin.py::compute_wireless_rates`](mec_into/src/digital_twin.py) |
| **(4)** | Wireless Uplink Delay | $d^w_{ij}(t) = \frac{z_{ij}(t) x_{ij}(t)}{\iota_{ij}(t)}$ | $z_{ij}\text{ (bits)}, d^w_{ij}\text{ (s)}$ | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(5)** | Energy Budget Constraint | $p_{ij}(t) d^w_{ij}(t) x_{ij}(t) \le Q^E_{ij}(t)$ | Stored energy vs transmission consumption | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(6)** | Battery Queue Evolution | $Q^E_{ij}(t+1) = Q^E_{ij}(t) + e_{ij}(t) - x_{ij}(t) p_{ij}(t) d^w_{ij}(t)$ | Harvested solar/RF arrival $e_{ij}\text{ (J)}$ | [`digital_twin.py::update_energy_buffers`](mec_into/src/digital_twin.py) |
| **(7)** | Peer Offloading Decision | $y^{i'}_{ij}(t) \in \{0, 1\}$ | Binary target selection | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(8)** | Wired Backhaul Delay | $d^c_{ij}(t) = \frac{z_{ij}(t) x_{ij}(t) y^{i'}_{ij}(t)}{C_{i'i}(t)}$ | $C_{i'i} = 100\text{ Mbps}$, zero if $i'=i$ | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(9)** | Computation Execution Delay| $d^m_{ij}(t) = \frac{z_{ij}(t) v_{ij}(t) x_{ij}(t)}{\sum_{i'=1}^N f^{i'}_{ij}(t)}$ | $v_{ij}\text{ (cycles/bit)}, f\text{ (cycles/s)}$ | [`environment.py::step`](mec_into/src/environment.py) |
| **(10)** | BS Computing Capacity | $\sum_{j=1}^{U_i} \sum_{i=1}^N f^{i'}_{ij}(t) \le F_{i'}(t), \forall i' \in \mathcal{N}$ | Server capacity limit $F_i = 5\text{ GHz}$ | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(11)** | System Income Reward | $r(t) = \sum_{i=1}^N \sum_{j=1}^{U_i} g_{ij}(t) x_{ij}(t) \cdot \mathbf{1}\{d_{ij}(t) \le \delta_{ij}(t)\}$ | $d_{ij} = d^w_{ij} + d^c_{ij} + d^m_{ij}$ | [`environment.py::step`](mec_into/src/environment.py) |
| **(12)** | Income Maximization Problem| $\max \sum_{t=0}^\infty \gamma^t r(t) \quad \text{s.t. } (5), (6), (10), C1-C4$ | Long-term discounted cumulative income | [`into_agent.py::INTOAgent`](mec_into/src/into_agent.py) |
| **(15)** | Nearest Neighbor Discrete Action | $a'(t) = \arg\min_{a(t) \in \mathcal{A}'(t)} \|a(t) - \hat{a}(t)\|$ | Discrete offloading affinity argmax | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(16)-(17)**| Energy Refinement Mapping | $\hat{x}_{ij}(t) = 0, \hat{y}^{i'}_{ij}(t) = 0$ if (5) violated | Feasibility enforcement | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(18)** | Minimum Required CPU | $l^{i'}_{ij}(t) = \frac{z_{ij}(t) v_{ij}(t) x_{ij}(t) y^{i'}_{ij}(t)}{\delta_{ij}(t) - d^w_{ij}(t) - d^c_{ij}(t)}$ | CPU needed to meet deadline | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **Priority** | Task Economic Priority | $\varrho_{ij}(t) = \frac{g_{ij}(t)}{l^{i'}_{ij}(t)}$ | Ratio of award to required computing capacity | [`action_refinement.py::refine_action`](mec_into/src/action_refinement.py) |
| **(19)** | Critic Loss & Parameter Update | $\theta^Q =: \theta^Q + \alpha_Q \mathbb{E}[(\rho - Q(s, a|\theta^Q)) \nabla_{\theta^Q} Q]$ | Bellman TD error optimization | [`into_agent.py::update`](mec_into/src/into_agent.py) |
| **(20)** | Deterministic Policy Gradient | $\theta^\phi =: \theta^\phi + \alpha_\phi \mathbb{E}[\nabla_{\theta^\phi} \phi(s) \nabla_a Q(s, a)|_{a=\phi(s)}]$ | Actor network gradient ascent | [`into_agent.py::update`](mec_into/src/into_agent.py) |
| **(21)-(22)**| Target Network Soft Updates | $\theta^{Q'} = \varepsilon_1 \theta^Q + (1-\varepsilon_1)\theta^{Q'}, \quad \theta^{\phi'} = \varepsilon_2 \theta^\phi + (1-\varepsilon_2)\theta^{\phi'}$ | Polyak smoothing ($\varepsilon_1=\varepsilon_2=0.001$) | [`into_agent.py::update`](mec_into/src/into_agent.py) |

---

## 3. Algorithm Implementation Details

### Algorithm 1: Fast Action Refinement
The continuous proto-action $\hat{a}(t) \in \mathbb{R}^{165 \times 4}$ generated by the actor network provides offloading preferences across discard and BS allocations. Algorithm 1 handles:
1. **Unconstrained Discrete Projection (Eq. 15):** Finds nearest discrete configuration $a'(t)$ via row-wise argmax over the 4 action slots.
2. **Device Battery Constraint (Eq. 5, 16, 17):** Computes wireless transmission delay $d^w_{ij}$ and evaluates required energy $p_{ij} d^w_{ij}$. If the device cannot afford the transmission, both $x_{ij}$ and $y_{ij}$ are set to 0.
3. **MEC Server Capacity Constraint (Eq. 10, 18):** For each BS $i'$, calculates the minimal CPU rate $l^{i'}_{ij}(t)$ needed to satisfy the remaining deadline $\delta_{ij}(t) - d^w_{ij}(t) - d^c_{ij}(t)$. Evaluates priority $\varrho_{ij}(t) = g_{ij}(t) / l^{i'}_{ij}(t)$. Tasks are sorted in ascending priority order; if total required CPU exceeds $F_{i'}(t)$, tasks with the lowest $\varrho_{ij}(t)$ are pruned until capacity feasibility is achieved.

### Algorithm 2: DRL-Based INTO
- **Actor Network:** 3-layer MLP with hidden layers `[3000, 1000]` activated by ReLU with Sigmoid output.
- **Critic Network:** 3-layer MLP with hidden layers `[1500, 600]` taking concatenated state and action representations.
- **Experience Replay Buffer:** Size $|\Omega| = 3000$. Mini-batch size = 64.
- **Exploration:** Gaussian noise $\eta(t) \sim \mathcal{N}(0, 0.1^2)$ added to continuous proto-actions during training.
- **Target Smoothing:** Soft updates with $\varepsilon_1 = \varepsilon_2 = 0.001$.

---

## 4. Evaluation and Baseline Comparison Summary

### Section V-A Parameter Analysis (Figures 3, 4, 5)
1. **Figure 3(a) — Learning Rate:** Evaluates $\alpha \in \{0.0001, 0.001, 0.01\}$. Smaller learning rate ($0.0001$) achieves the highest steady-state average income by avoiding sub-optimal oscillations.
2. **Figure 3(b) — Discount Factor:** Evaluates $\gamma \in \{0.59, 0.79, 0.99\}$. Larger discount factor ($\gamma = 0.99$) affords greater foresight over long-term battery replenishment and achieves the highest cumulative income.
3. **Figure 4(a) — Task Award:** As BS-1 task awards scale by $\xi \in [1.0, 3.0]$, INTO prioritizes BS-1 tasks; offloaded tasks in BS-1 rise while BS-2 and BS-3 decrease.
4. **Figure 4(b) — Processing Density:** As processing density of BS-2 and BS-3 scales by $\zeta \in [1.0, 1.4]$, CPU demand in BS-2/3 increases, leading INTO to offload more BS-1 tasks while curtailing lower-yield BS-2/3 tasks.
5. **Figure 5(a) — Channel Gain:** As channel gain scale $\sigma$ increases from $0.2$ to $1.2$, wireless transmission delay and energy shrink, allowing more tasks to be admitted until MEC computing capacity limits are reached.
6. **Figure 5(b) — Harvested Energy Arrival Rate:** As energy arrival rate scale $\rho$ increases from $0.2$ to $1.2$, device batteries are replenished faster, admitting more offloading candidates until CPU capacity becomes the bottleneck.

### Section V-B Baseline Comparison (Figure 6)
Across all 4 evaluation dimensions, INTO consistently outperforms OLNPO, LPWPO, and DDQN:
- **Figure 6(a) (Task Data Size $\chi = 1.0 \dots 1.6$):** Larger data sizes increase transmission and computing delays; income monotonically decreases across all schemes. INTO maintains the highest income due to joint global optimization.
- **Figure 6(b) (MEC Capacity $F_i = 5.0 \dots 8.0\text{ GHz}$):** Higher computing capacities allow more tasks to be admitted before deadlines expire. INTO achieves higher income than LPWPO (which only offloads leftover capacity locally) and OLNPO (which cannot share loads).
- **Figure 6(c) (Task Delay Tolerance $\psi = 1.0 \dots 4.0$):** Looser deadlines allow lower per-task CPU allocations, enabling BSs to admit more tasks.
- **Figure 6(d) (Channel Bandwidth $B = 0.6 \dots 1.6\text{ MHz}$):** Larger bandwidth reduces transmission latency, leaving more time for computing.

---

## 5. Artifact Directory & Output Inventory

All generated artifacts, source code, checkpoints, and figures are organized as follows:

```text
├── run_experiments.py                   # Root execution script
├── requirements.txt                     # Pinned dependencies
├── reports/
│   ├── final_execution_report.md        # Technical execution and equation mapping report
│   └── task_generation_and_distribution.md # Workload modeling and statistical summary
├── outputs/
│   ├── checkpoints/                     # Model weights (.pt)
│   ├── plots/                           # High-res reproduction figures (PNG)
│   └── results/                         # Raw numerical experimental results (CSV)
└── mec_into/
    ├── config/experiment_config.yaml    # Exact simulation parameters
    ├── main.py                          # CLI orchestrator
    └── src/
        ├── action_refinement.py         # Algorithm 1: Fast Action Refinement
        ├── digital_twin.py              # Digital Twin state mirror & SINR
        ├── environment.py               # EdgeSimPy MEC environment
        ├── evaluation.py                # Parameter & baseline suites
        ├── into_agent.py                # Algorithm 2: INTO DDPG agent
        ├── models.py                    # Actor, Critic, DDQN DNNs
        ├── task_generator.py            # Runtime task sampling
        ├── training.py                  # INTO & DDQN training loops
        └── baselines/
            ├── ddqn.py                  # DDQN benchmark
            ├── lpwpo.py                 # LPWPO benchmark
            └── olnpo.py                 # OLNPO benchmark
```
