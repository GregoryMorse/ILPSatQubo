# ILPSatQubo

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3631908.3631929-blue.svg)](https://doi.org/10.1145/3631908.3631929)
[![Journal](https://img.shields.io/badge/Journal-ACM%20ICACS%202023-00599C.svg)](https://www.icacs.org/2023.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official code supplement for the published conference paper:**  
*On Optimal QUBO Encoding of Boolean Logic, (Max-)3-SAT and (Max-)k-SAT with Integer Programming*  
Gregory Morse, Tamás Kozsik  
ICACS '23 — ACM, DOI: [10.1145/3631908.3631929](https://doi.org/10.1145/3631908.3631929)

Gregory Morse — [gregory.morse@live.com](mailto:gregory.morse@live.com)

---

## Abstract

We present an asymptotic improvement in the number of variables ($n + m\lfloor\log_2(k-1)\rfloor$) required for state-of-the-art formulation of (max-)k-SAT problems when encoded as a Quadratic Unconstrained Binary Optimization (QUBO) problem. We further show a variable reduction technique for (max-)3-SAT formulae which achieves the optimal number of substitution variables. We show optimality empirically by presenting an Integer Linear Programming (ILP) construction for arbitrary Boolean formulas. We show how various goals can be encoded, and this model can be generalised to searching for arbitrary substitution variable functions. Lastly, we show the optimal high-order substitution reduction in cubic QUBO equations which has smaller coefficients than the typical construction used.

---

## Conference Presentation

This work was presented at the **International Conference on Algorithms, Computing and Systems (ICACS) 2023**.

- **Authors:** Gregory Morse and Tamás Kozsik
- **Paper:** *On Optimal QUBO Encoding of Boolean Logic, (Max-)3-SAT and (Max-)k-SAT with Integer Programming*
- **Conference:** ICACS 2023
- **Location:** Larissa, Greece (conference changed to virtual format)
- **Presentation time:** October 25, 2023, 15:00
- **Conference page:** https://www.icacs.org/2023.html

---

## Contents

| File | Description |
|---|---|
| `ilp_sat_qubo.py` | Main implementation: ILP encoders, QUBO construction, solver interfaces, benchmarking, and plot generation |

---

## Methodology

### QUBO Penalty Functions

A Boolean function $f : \{0,1\}^n \to \{0,1\}$ is encoded as a QUBO penalty $P(x_1,\ldots,x_n,s_1,\ldots,s_k)$ such that:

$$P \;=\; 0 \iff f(x_1,\ldots,x_n) = 1, \qquad P > 0 \text{ otherwise.}$$

The penalty polynomial is at most quadratic (required by QUBO) and is constructed via auxiliary substitution variables $s_1,\ldots,s_k$ representing sub-functions of the original variables.

### ILP Formulation

The QUBO coefficients are determined by solving an ILP whose constraints enforce the penalty conditions over all truth-table rows and all valid / invalid substitution assignments.  Two main solver paths are provided:

- **`ilp_ideal_qubo_encoding`** — Given explicit substitution sub-functions, solves for optimal coefficients via `scipy.optimize.linprog` (HiGHS MIP), CPLEX (via `docplex`), Gurobi, Z3, or GLPK.
- **`ilp_ideal_qubo_encoding_allsubs`** / **`iqp_ideal_qubo_encoding_allsubs`** — Searches jointly over substitution function and coefficient, using ILP with McCormick / Fortet linearisation of bilinear terms.
- **`find_ideal_qubo_encoding_allsubs_tree`** — Tree-search enumeration of substitution candidates combined with incremental ILP feasibility checks.

### Encoding Pipeline for k-SAT

The function `max_ksat_to_qubo` converts a CNF formula to a QUBO dictionary:

1. Clauses of length > 3 are reduced recursively (Nüßlein-style or via `cnf_linalg_solve`) to at most 3-literal clauses with logarithmically many ancilla bits.
2. Groups of 3-literal clauses sharing a common 2-variable sub-clause are identified via `find_three_sat_common_two` (backed by the Set-Trie) and encoded jointly with a single shared substitution variable.
3. Remaining 1-, 2-, and 3-literal clauses are encoded with the optimal per-type QUBO formulas derived by the ILP engine and stored in `all_ternary_qubo`.

### Comparison Baseline

The Nüßlein (2023) encoding is included as `nusslein=True` for direct comparison of auxiliary variable count and physical qubit requirements on D-Wave Advantage (Pegasus topology).

---

## Requirements

### Core (required)

```
pip install numpy scipy
```

### Optional Solvers

| Package | Purpose |
|---|---|
| `gurobipy` | Gurobi MIP solver (free academic licence available) |
| `cplex` / `docplex` | IBM ILOG CPLEX (free academic licence available) |
| `z3-solver` | Microsoft Z3 SMT/CP solver |
| `glpk` | GNU Linear Programming Kit |

### Quantum Annealing (optional)

```
pip install dimod minorminer dwave-system dwave-qbsolv
```

A D-Wave Leap token must be set in the environment variable `DWAVE_API_TOKEN`.

### Set-Trie (optional, required for clause-group covering in `max_ksat_to_qubo`)

```
pip install pysettrie
```

`pysettrie` is a Cython-accelerated set-trie library maintained by the same author at [github.com/GregoryMorse/pysettrie](https://github.com/GregoryMorse/pysettrie). The local `settrie.pyx` / `makesettrie.pyx` sources in this repository are the legacy predecessor and are no longer needed.

---

## Installation

```bash
git clone https://github.com/GregoryMorse/ILPSatQubo.git
cd ILPSatQubo
pip install numpy scipy          # minimum required packages
# (optional) pip install pysettrie           # set-trie for clause-group covering
# (optional) pip install gurobipy docplex z3-solver glpk
# (optional) pip install dimod minorminer dwave-system dwave-qbsolv
```

---

## Usage

The module is designed to be imported or edited interactively; uncomment the relevant `print(...)` calls for the experiment of interest.  Representative entry points:

```python
from ilp_sat_qubo import (
    ilp_ideal_qubo_encoding,
    ilp_ideal_qubo_encoding_allsubs,
    max_ksat_to_qubo,
    random_k_sat,
    get_ternary_func,
    nary_func_to_num,
)

# --- Optimal QUBO encoding of a 3-literal SAT clause (no auxiliary variable) ---
f = lambda a, b, c: a | b | c          # a ∨ b ∨ c
print(ilp_ideal_qubo_encoding(f, 3, [[]], coeff=None))

# --- Encode a random 3-SAT instance as a QUBO ---
cnf = random_k_sat(n=20, clauses=85, k=3)
varmap = {i+1: f'x{i}' for i in range(20)}
bqm, const, varmap, submap = max_ksat_to_qubo(cnf, varmap, nusslein=False)
# bqm is a dict {(var_i, var_j): coeff} ready for dimod / D-Wave

# --- Search for optimal substitution function + coefficients ---
result = ilp_ideal_qubo_encoding_allsubs(lambda a, b, c, d: a | b | c | d, n=4, nsubs=1)
print(result)
```

### Reproducing Paper Results

The function `all_ternary_qubo_encoding()` (called at module load time and cached in `all_ternary_qubo`) enumerates optimal QUBO encodings for all 256 ternary Boolean functions and prints LaTeX-formatted tables of coefficients and substitution expressions matching the paper's tables.

The function `check_random_three_sat()` reproduces the empirical benchmarks (substitution variable count, non-zero couplings, physical qubit count on D-Wave, clause-satisfaction rate) for the random 3-SAT and 8-SAT experiments reported in the paper.

The function `random_sat_plots()` regenerates the figures comparing the new encoding against the Nüßlein (2023) baseline.

---

## Key Functions Reference

| Function | Description |
|---|---|
| `ilp_ideal_qubo_encoding(f, n, subs, coeff)` | Solve for QUBO coefficients given sub-functions |
| `ilp_ideal_qubo_encoding_allsubs(f, n, nsubs)` | Joint search over substitution & coefficients (ILP) |
| `iqp_ideal_qubo_encoding_allsubs(f, n, nsubs)` | Same via IQP (integer quadratic programs) |
| `find_ideal_qubo_encoding_allsubs_tree(f, n, nsubs)` | Tree-search with incremental ILP |
| `max_ksat_to_qubo(cnf, varmap, nusslein)` | Convert k-SAT CNF to QUBO dictionary |
| `max_three_sat_to_qubo(cnf, varmap, nusslein)` | Specialised pipeline for 3-SAT |
| `cnf_linalg_solve(n)` | Closed-form coefficient sets for n-literal clauses |
| `all_ternary_funcs()` | Symbolic representation of all 256 ternary functions |
| `random_k_sat(n, clauses, k)` | Random k-SAT instance generator |
| `check_random_three_sat()` | Empirical benchmark loop (D-Wave + QBSolv) |
| `random_sat_plots()` | Plot generation for benchmark figures |

---

## Citing

If you use this code or build upon this work, please cite the paper:

**ACM Reference Format:**

> Gregory Morse and Tamás Kozsik. 2024. On Optimal QUBO Encoding of Boolean Logic, (Max-)3-SAT and (Max-)k-SAT with Integer Programming. In *Proceedings of the 7th International Conference on Algorithms, Computing and Systems (ICACS '23)*. Association for Computing Machinery, New York, NY, USA, 145–153. https://doi.org/10.1145/3631908.3631929

**BibTeX:**

```bibtex
@inproceedings{morse2024qubo,
  author    = {Morse, Gregory and Kozsik, Tam{\'{a}}s},
  title     = {On Optimal {QUBO} Encoding of Boolean Logic, ({Max-})3-{SAT}
               and ({Max-})k-{SAT} with Integer Programming},
  booktitle = {Proceedings of the 7th International Conference on Algorithms,
               Computing and Systems},
  series    = {ICACS '23},
  pages     = {145--153},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  year      = {2024},
  month     = feb,
  doi       = {10.1145/3631908.3631929},
  isbn      = {9798400709098},
  url       = {https://doi.org/10.1145/3631908.3631929}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Gregory Morse — [gregory.morse@live.com](mailto:gregory.morse@live.com)  
GitHub: [https://github.com/GregoryMorse](https://github.com/GregoryMorse)


