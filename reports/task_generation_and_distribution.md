# Task Generation and Distribution Summary

**Reference Paper:** *Digital Twin-Driven Intelligent Task Offloading for Collaborative Mobile Edge Computing* (IEEE JSAC 2023)  
**Authors:** Yongchao Zhang, Jia Hu, and Geyong Min  
**Implementation Package:** `mec_into`  
**Execution Environment:** EdgeSimPy + PyTorch DRL + Digital Twin Mirror  

---

## 1. Task Generation Process

In the simulated collaborative Mobile Edge Computing (MEC) system, computation tasks are generated programmatically at the beginning of each discrete time slot $t$ ($t = 0, 1, 2, \dots$). 

- **Timing of Generation:** In accordance with Section III-A of the research paper, each Mobile Device (MD / IoT device) generates exactly one new computation task $O_{ij}(t)$ at the onset of each time slot $t$.
- **Number of Tasks Generated:** Across the 3 collaborative Base Stations (BSs), there are $U_1 = 30$, $U_2 = 55$, and $U_3 = 80$ devices respectively, yielding a total of:
  $$M = \sum_{i=1}^{N} U_i = 30 + 55 + 80 = 165 \text{ tasks per time slot}$$
- **Lifecycle & Regeneration:** As specified in Section III-A, computation tasks are delay-sensitive with maximum tolerance delay $\delta_{ij}(t) \le \tau$ (where $\tau = 1.0\text{ s}$ denotes the time slot duration). Consequently, tasks are evaluated, executed or discarded within the time slot; uncompleted tasks do not persist beyond their deadlines, and a fresh independent task profile is generated at every new time slot $t$. No static or pre-recorded task trace is reused.

---

## 2. Task Parameter Distributions

Task parameters are sampled at runtime using strictly bounded pseudo-random distributions directly transcribed from Section V-A and Table II of the reference paper:

| Task Parameter | Notation | Probability Distribution | Parameter Range | Physical Unit | Source in Paper |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Task Data Size** | $z_{ij}(t)$ | Uniform $U[50, 80]$ | $50.0 - 80.0$ | kbit ($50 \times 10^3 - 80 \times 10^3$ bits) | Section V-A, p. 3041 |
| **Processing Density** | $v_{ij}(t)$ | Uniform $U[3000, 8000]$ | $3,000 - 8,000$ | CPU cycles / bit | Section V-A, p. 3041 |
| **Maximum Delay Tolerance**| $\delta_{ij}(t)$ | Uniform $U[0.2, 2.0]$ | $0.2 - 2.0$ | seconds | Section V-A, p. 3041 |
| **Task Completion Award** | $g_{ij}(t)$ | Uniform $U[1, 3]$ | $1.0 - 3.0$ | monetary / reward unit | Section V-A, p. 3041 |
| **Device Transmit Power** | $p_{ij}(t)$ | Uniform $U[50, 180]$ | $50.0 - 180.0$ | mW ($0.05 - 0.18$ Watts) | Section V-A, p. 3041 |
| **Harvested Energy Arrival**| $e_{ij}(t)$ | Uniform $U[1.0, 1.5]$ | $1.0 - 1.5$ | mW / mJ (per slot $\tau = 1\text{s}$) | Section V-A, p. 3041 |
| **Channel Bandwidth** | $B$ | Fixed Nominal | $1.0$ | MHz ($10^6$ Hz) | Table II, Section V-A |
| **Noise Power Spectral Density**| $N_0$ | Fixed Nominal | $1.0 \times 10^{-8}$ | W/Hz | Table II, Section V-A |
| **Inter-BS Transmission Rate** | $C_{i'i}(t)$ | Fixed Nominal | $100$ | Mbps ($10^8$ bps) | Section III-A, Table II |
| **MEC Server CPU Capacity** | $F_i(t)$ | Fixed Nominal | $5.0$ | GHz ($5.0 \times 10^9$ cycles/s) | Section V-A, Table II |

---

## 3. Distribution Across Base Stations

The 165 IoT devices are geographically partitioned into non-overlapping base station coverage domains matching the heterogeneous load profile established in the literature and Section V-A:

- **BS-1 (Light Load):** 30 IoT devices (Device indices `0` to `29`)
- **BS-2 (Medium Load):** 55 IoT devices (Device indices `30` to `84`)
- **BS-3 (Heavy Load / Congested Hotspot):** 80 IoT devices (Device indices `85` to `164`)

Every device $j \in \mathcal{U}_i$ is permanently associated with its serving BS $i$ for uplink radio transmission over orthogonal bandwidth allocations. Tasks can subsequently undergo peer offloading over high-speed wired links from serving BS $i$ to peer BS $i' \in \mathcal{N} \setminus \{i\}$.

---

## 4. Random Generation Implementation

All stochastic sampling is implemented in `mec_into/src/task_generator.py` via `numpy.random.RandomState`.

- **Reproducibility & Seeding:** The pseudo-random generator accepts an explicit integer seed (`seed=42` by default for training; distinct seeds `123` and `999` for evaluation suites). This ensures repeatable sequences of channel gains, task sizes, deadlines, and awards across independent experimental iterations.
- **Independence:** Independent uniform streams sample $z_{ij}(t)$, $v_{ij}(t)$, $\delta_{ij}(t)$, $g_{ij}(t)$, $p_{ij}(t)$, and $e_{ij}(t)$ for each device, while channel gain $h_{ij}(t)$ incorporates Rayleigh small-scale multipath fading combined with nominal path-loss.

---

## 5. Task Assignment and Offloading Decisions

Distinction between task generation, admission, offloading target selection, and execution:

1. **Workload Generation:** An IoT device $j$ under BS $i$ samples task $O_{ij}(t)$.
2. **Actor Proto-Action Formulation:** The DRL policy outputs continuous affinity vector $\hat{a}(t) \in \mathbb{R}^{165 \times 4}$, where columns correspond to `[Discard, Process at BS-1, Process at BS-2, Process at BS-3]`.
3. **Nearest Neighbor Mapping (Eq. 15):** The argmax maps continuous affinity to unconstrained discrete choices $x'_{ij} \in \{0, 1\}$ and $y'^{i'}_{ij} \in \{0, 1\}$.
4. **Energy Feasibility Filter (Eq. 5, 16, 17):**
   $$p_{ij} d^w_{ij}(t) \le Q^E_{ij}(t)$$
   If the device battery cannot sustain transmission, the task is dropped ($x_{ij} = 0$).
5. **Computing Resource & Priority Refinement (Algorithm 1, Eq. 10, 18):**
   The minimum CPU required to meet the deadline is computed as:
   $$l^{i'}_{ij}(t) = \frac{z_{ij}(t) v_{ij}(t)}{\delta_{ij}(t) - d^w_{ij}(t) - d^c_{ij}(t)}$$
   Each task's economic density is evaluated as $\varrho_{ij}(t) = \frac{g_{ij}(t)}{l^{i'}_{ij}(t)}$. If a base station's total CPU demand exceeds $F_{i'}(t)$, tasks with the lowest $\varrho_{ij}(t)$ are sequentially discarded until the capacity constraint is strictly satisfied.
6. **Baseline Policies:**
   - **OLNPO:** Strict local constraint $y^{i'}_{ij} = 0$ for all $i' \ne i$. BS-3 drops high-value tasks due to local overload while BS-1 remains idle.
   - **LPWPO:** Serves local tasks first; only spare remaining CPU capacity is allocated to peer-offloaded tasks.
   - **DDQN:** Employs value-based Q-learning directly over the discrete combinatorial action space.

---

## 6. Parameter Scaling in Experiments

To reproduce Section V-A (Figures 4, 5) and Section V-B (Figure 6), controlled scale factors modulate specific parameters while keeping all others frozen at baseline values:

| Experiment | Modulated Parameter | Scale Factor Symbol | Range & Step | Affected Scope | Fixed Parameters |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Fig. 4(a)** | Task Award | $\xi$ | $[1.0, 3.0]$, step $0.5$ | BS-1 devices only | BS-2/3 awards, sizes, densities, channels |
| **Fig. 4(b)** | Processing Density | $\zeta$ | $[1.0, 1.4]$, step $0.1$ | BS-2 & BS-3 devices only | BS-1 density, awards, sizes, channels |
| **Fig. 5(a)** | Channel Power Gain | $\sigma$ | $[0.2, 1.2]$, step $0.2$ | All 165 IoT devices | Awards, densities, sizes, batteries |
| **Fig. 5(b)** | Harvested Energy Rate | $\rho$ | $[0.2, 1.2]$, step $0.2$ | All 165 IoT devices | Awards, densities, sizes, channels |
| **Fig. 6(a)** | Task Data Size | $\chi$ | $[1.0, 1.6]$, step $0.1$ | All 165 IoT devices | Capacities ($5\text{ GHz}$), Bandwidth ($1\text{ MHz}$) |
| **Fig. 6(b)** | MEC Server Capacity | $F_i$ | $[5.0, 8.0]\text{ GHz}$, step $0.5$ | All 3 Base Stations | Task sizes, awards, bandwidth ($1\text{ MHz}$) |
| **Fig. 6(c)** | Delay Tolerance | $\psi$ | $[1.0, 4.0]$, step $0.5$ | All 165 IoT devices ($\delta \sim U[0.2\psi, 2]$) | Sizes, awards, capacities ($5\text{ GHz}$) |
| **Fig. 6(d)** | Channel Bandwidth | $B$ | $[0.6, 1.6]\text{ MHz}$, step $0.2$ | System-wide RF bandwidth | Capacities ($5\text{ GHz}$), Task sizes ($1.0\chi$) |

---

## 7. Implementation-to-Paper Mapping

| Paper Specification | Implemented Code Element | Verification Status | Notes & Assumptions |
| :--- | :--- | :---: | :--- |
| $N = 3, U_1=30, U_2=55, U_3=80$ | `experiment_config.yaml` lines 5-6 | Exact Match | Total 165 IoT devices |
| $z_{ij} \sim U[50, 80]\text{ kbit}$ | `task_generator.py::generate_tasks_for_slot` | Exact Match | Uniform bit sampling |
| $v_{ij} \sim U[3000, 8000]\text{ cycles/bit}$| `task_generator.py::generate_tasks_for_slot` | Exact Match | Call-graph complexity model |
| $\delta_{ij} \sim U[0.2, 2.0]\text{ s}$ | `task_generator.py::generate_tasks_for_slot` | Exact Match | Strict deadline enforcement |
| $g_{ij} \sim U[1, 3]$ | `task_generator.py::generate_tasks_for_slot` | Exact Match | Linear reward contribution |
| $p_{ij} \sim U[50, 180]\text{ mW}$ | `task_generator.py::generate_tasks_for_slot` | Exact Match | Transmit power bound |
| $e_{ij} \sim U[1, 1.5]\text{ mW}$ | `task_generator.py::generate_tasks_for_slot` | Exact Match | Energy arrival per slot |
| $B = 1\text{ MHz}, N_0 = 10^{-8}\text{ W/Hz}$| `digital_twin.py::compute_wireless_rates` | Exact Match | Shannon capacity formulation |
| $C_{i'i} = 100\text{ Mbps}$ | `action_refinement.py`, `environment.py` | Exact Match | Fiber optical backhaul mesh |
| $F_i = 5\text{ GHz}$ | `experiment_config.yaml`, `digital_twin.py` | Exact Match | Base server capacity |

---

## 8. Relevant Code References

- **Task generation & stochastic sampling:** `mec_into/src/task_generator.py`
- **Digital Twin state mirror & SINR rates:** `mec_into/src/digital_twin.py`
- **Action refinement & priority scheduling:** `mec_into/src/action_refinement.py`
- **Execution environment & EdgeSimPy components:** `mec_into/src/environment.py`
- **Parameter sweeps & evaluation suites:** `mec_into/src/evaluation.py`
