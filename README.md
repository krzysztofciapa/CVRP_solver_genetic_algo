# High-Performance CVRP Solver: Structural Reasoning & Heterogeneous Island Model

Welcome to my CVRP (Capacitated Vehicle Routing Problem) solver! 

This project explores how optimization algorithms can acquire reasoning capabilities and recover structural coherence from highly noisy data under strict computational budgets. 

### Problem Definition: Shuffled Global Permutation
In standard CVRP, `.lcvrp` files typically provide near-ideal global permutations ("giant tours") that can be mathematically partitioned into optimal routes in $O(n^2)$ using Bellman-Ford or in $O(n)$ using Thibaut Vidal's Split algorithm. 

However, this solver is engineered for a much harsher environment (version 3 of the competition package). Upon loading, the input permutation is deliberately shuffled, yielding a geometrically unordered and low-quality visit sequence. The main research challenge was to adapt the $O(n)$ Split algorithm and construct an architecture capable of explicitly reasoning through this geometric chaos to reconstruct a mathematically optimal visit sequence.

---

##  Performance

Because the absolute global optimums for this specific shuffled variant are unknown, performance is evaluated through relative empirical benchmarking. 

This architecture achieved state-of-the-art results across major benchmark datasets. **My solver found the absolute best-known solutions for 5 out of 5 core dataset classes displayed on the leaderboard**, systematically outperforming other participants:

| Dataset / Benchmark | My Best Cost | Runner-up | Advantage |
| :--- | :--- | :--- | :--- |
| **Vrp-Set-XXL / Leuven2** | **330597.26** | 331498.60 | Outperformed |
| **Vrp-Set-D / ORTEC-n323**| **304434.00** | 306716.00 | Outperformed |
| **Vrp-Set-X / X-n209-k16** | **42158.46** | 42725.73 | Outperformed |
| **Vrp-Set-XML100** | **18937.56** | 18937.56 | Tied for Best |
| **Vrp-Set-P / P-n19-k2** | **371.27** | 371.27 | Tied for Best |

---

## Architecture: The Heterogeneous Island Model

To maximize CPU utilization and prevent premature convergence, the solver implements a multi-threaded Heterogeneous Island Model. Each island operates on a dedicated thread and assumes a distinct operational role:

* **Explorers (Even IDs):** Tasked with high-variance exploration of new, promising areas. They utilize larger population sizes, high mutation probabilities, and fast VND operations (Relocate + Swap).
* **Exploiters (Odd IDs):** Focused on deep local reasoning and uncovering deep optima. They use lower mutation rates but apply computationally heavy VND operations (Ejection Chains, Path Relinking, 3-Swap, 4-Swap).

### Diversity Management & Asynchronous Migration
* **Catastrophic Mutation & Restarts:** Population diversity is strictly monitored using the **Broken Pairs Distance (BPD)** metric. Stagnation triggers a catastrophic mutation. If an island fails to recover after 5 unsuccessful catastrophes, it transfers its best individuals elsewhere, wipes its population to restart completely from scratch, and blocks incoming migration for 60 seconds.
* **Migration & Broadcasts:** Gene exchange is asynchronous and depends on the specific island's health. Aside from fitness, an adequate structural difference in the BPD metric is a strict prerequisite for any individual entering an island. Additionally, a broadcast mechanism instantly propagates newly discovered global bests across the entire archipelago.

---

##  Genetic Operators & Noise Recovery

The search process is driven by **Adaptive Operator Selection (AOS)**, dynamically evaluating historical accuracy to allocate compute budget only to the most effective operators.

### 1. Crossover (Recombination)
* **Geometric (Neighbor-based):** Selects a random client from Parent 1, assigns the route/group IDs of its nearest neighbors to the offspring, and inherits the remaining assignments from Parent 2.
* **Sequence Routing Crossover (SREX):** Exchanges entire routes. Approximately half the routes are inherited from Parent 1, followed by as many non-conflicting routes as possible from Parent 2. Remaining conflicting clients are inserted using a Regret-3 heuristic.

### 2. Mutations
* **MicroSplit:** Executes the optimal Split algorithm within a smaller permutation window, allowing it to isolate a structurally "good" segment and partition it perfectly.
* **MergeSplit:** Merges two routes and performs an optimal Split on their clients. Highly effective for shuffling and re-splitting when permutation ranks become deadlocked.
* **Ruin & Recreate (RR):** Removes a client and its nearest neighbors (amount based on disruption intensity), then reinserts them factoring in route capacities and geometric proximity to other routes.
* **Adaptive LNS (ALNS):** A dynamic RR variant where the destruction strategy is dynamically selected to target specific weaknesses: permutation clusters, over-capacitated routes, or routes with the worst distance and fill rate.
* **ReturnMinimizer:** Eliminates truck returns by reducing over-capacitated routes (e.g., squashing 130% fill rate down to 100%).
* **EliminateReturns:** Manipulates packing density (fill rate). It removes the tail ends of routes where vehicles deadhead (return empty, utilization < 70%). From the evaluator's perspective, one route packed at 200% yields the exact same fitness result as two 100% vehicles, provided the segments do not overlap.

### 3. Deep Local Search (VND)
* **Variable Neighborhood Descent (VND):** Evaluates client relocations and cross-route swaps. Feasible safe moves strictly use fast delta evaluation. Infeasible/unsafe moves strictly simulate the cost of the entire route.
* **Path Relinking (PR):** Calculates topological differences between an individual and the island's guide solution (best individual), attempting to connect them to find a superior intermediate configuration.
* **Ejection Chains:** Executes complex, chained sequences of client displacements and swaps.
* **3-Swap / 4-Swap:** Selects geometrically close clients and evaluates all combinations ($3!$ and $4!$ moves are computationally acceptable here).

---

##  Advanced Compute & Memory Architecture

To explicitly manage strict compute budgets and hardware limitations, the system bypasses standard library overheads in favor of raw C++ optimizations:

### Hardware-Aware Distance Evaluation
The `ThreadSafeEvaluator` dynamically manages L1/L3 CPU cache constraints based on graph complexity. For instances under `MATRIX_THRESHOLD = 5000`, it relies on a flattened, SIMD-friendly contiguous $1D$ distance matrix. For massive instances where a $25 \times 10^6$ double matrix would cause catastrophic cache thrashing, the system bypasses the matrix entirely and computes Euclidean distances on-the-fly.

### Dual-Layer 64/32-bit Hash Deduplication
To completely eliminate redundant full-route simulations during VND, the `LocalCache` employs a highly optimized bitwise memory pool (`1 << 25` entries, approx. 33.5 million slots). To guarantee $0\%$ false-positive rates on sequence evaluations, it uses a dual-verification mechanism:
1. **64-bit Signature:** Computes an XOR-based hash utilizing a golden ratio constant (`0x9e3779b97f4a7c15ULL`) to mitigate value collisions.
2. **32-bit Position Checksum:** Simultaneously validates the exact topological sequence by weighting values by their positional index.

### Mathematical Adaptive Operator Selection (AOS)
Compute budget isn't allocated blindly. The `AdaptiveOperatorSelector` routes CPU time using an Exponential Moving Average (`DECAY_FACTOR = 0.95`). An operator's execution probability is determined by a hybrid mathematical score combining its binary success rate and its normalized relative fitness improvement: `score = success_rate * (1.0 + norm_improvement)`. This ensures dynamic adjustment to the landscape's topology, heavily prioritizing specific destruction strategies (like `WORST_ROUTES` or `TARGET_OVERFLOW`) only when mathematically justified.

### RoutePool & Beam Search (MIP Inspired)
Inspired by Column Generation in Mixed Integer Programming, a global `RoutePool` caches up to `10,000` of the absolute best singular routes. Periodically, a **Beam Search** (with `BEAM_WIDTH = 50`) scans this immense pool to construct mathematically coherent, non-overlapping "Frankenstein" states. These structures are force-injected into stagnant populations, transferring high-quality deep representations across search epochs.
### Diverse Initialization Methods
To guarantee high initial diversity despite the shuffled input, multiple initialization routines are deployed:
* **Random & Chunked** generation.
* **Explicit K-Center:** Deliberately avoids using centroids (unlike K-Means), making it strictly viable for explicit cost matrix instances.
* **Double Bridge Split:** Creates a temporary disrupted permutation using a Double Bridge move, which is subsequently passed through the Split algorithm to ensure the initial starting routes maintain rigorous mathematical coherence.

---

##  Build and Reproducibility

The project is structured as a Visual Studio Solution, optimized for C++ performance and multi-threading.

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   
2. **Build using MSBuild: Note: Compiling in the Release configuration is mandatory to accurately evaluate the algorithm's performance and accuracy-vs-compute metrics.**
   ```bash
      msbuild LcVRPContest.sln /p:Configuration=Release

3.**   Run an experiment:
Execute the solver against any specific dataset instance:**
  ```bash
    Release\LcVRPContest.exe data/lcvrp/Vrp-Set-XXL/Leuven2.lcvrp

