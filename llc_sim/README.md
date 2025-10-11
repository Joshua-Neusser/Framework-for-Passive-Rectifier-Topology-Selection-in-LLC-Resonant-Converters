## 📦 LLC Simulator Library
---
A Python library for simulating LLC resonant converters using [PySpice](https://pyspice.fabrice-salvaire.fr/).  
It acts as a high-level wrapper around native PySpice syntax to improve **readability**, **modularity**, and **reusability** in code that involves LLC converters.

---

### 🚀 Installation
Run the following command on the main directory.

```bash
pip install -e ./llc_sim

```
---

## 📘 Documentation

This library provides three main functions for simulating LLC resonant converters using PySpice:

---

### `build_llc_circuit(params)`

Builds a complete LLC converter circuit.

---

### `simulate(circuit, fsw, TimeStep, SimCycles, TimeStartSave=0, show_node_names=False)`

Runs a transient simulation on the circuit.

- `circuit`: Built with `build_llc_circuit`
- `TimeStep`: Simulation step size
- `SimCycles`: Number of switching cycles
- `show_node_names`: (optional) Print node names to console

---

### `analyze(analysis, fsw, TimeStep, SimCycles)`

Returns a dictionary with key performance results:

- `voutAVG`, `iSecRMS`, `iRPK`, `iDSPK`, `vcsRMS`, etc.

---

### ▶️ Example

```python
from llc_Sim import build_llc_circuit, simulate, analyze

params = {
    'fsw': 90e3,
    'Vbus': 400,
    'Cs': 6.8e-9,
    'Ls': 160e-6,
    'Lm': 750e-6,
    'Co': 10e-6,
    'Rload': 100,
    'n': 3.25
}

circuit = build_llc_circuit(params)
analysis = simulate(circuit, params['fsw'], 50e-9, 1500)
results = analyze(analysis, params['fsw'], 50e-9, 1500)
