# pybbn — Bayesian Belief Network Assurance Case Tool

A robotics safety assurance tool that uses **Bayesian Belief Networks (BBN)** to probabilistically verify whether a robot mission meets its requirements. Visualizes the assurance case as a **Goal Structuring Notation (GSN)** diagram and provides an interactive GUI for real-time belief tuning.

---

## What It Does

Given a robot mission (e.g. navigate terrain, avoid collisions, maintain pose accuracy), this tool:

1. **Builds a BBN** — models each requirement as a probabilistic node with conditional probability tables (CPTs)
2. **Runs inference** — propagates probabilities through the network using the PPTC algorithm
3. **Generates a GSN assurance case** — exports the belief network as a YAML-based Goal Structuring Notation diagram (via `gsn2x`) rendered as a PNG
4. **Launches an interactive GUI** — lets you tune input probabilities and thresholds via sliders and see live belief updates in bar charts and tables

### Example mission modelled

```
Meeting Requirements (GoalNode)
├── Robot Nav Terrain under Threshold (MinThresholdNode)
│   └── P(robot on navigable terrain) (SuccessNode)  ← tunable slider
├── Robot Collision under Threshold (MaxThresholdNode)
│   └── P(robot not collide) (SuccessNode)            ← tunable slider
└── Robot Pose under Threshold (MinThresholdNode)
    └── P(robot pose within region) (SuccessNode)     ← tunable slider
```

---

## Architecture

```
pybbn/
├── assurance_case.py   # Mission definition — builds the BBN with nodes/edges
├── bbn.py              # BBN class — wraps pybbn library, inference, GSN export
├── doe.py              # Node type definitions (GoalNode, SuccessNode, ThresholdNode variants)
├── gui.py              # customtkinter GUI — sliders, bar charts, GSN image, table
├── ctksliders.py       # Scrollable slider frame widget
├── ctktable.py         # Custom table widget
├── custom_sliders.py   # Slider helpers
├── helper.py           # Utility functions
├── logger.py           # Logging setup (console=INFO, file=DEBUG)
├── test.py             # Basic tests
├── requirements.txt    # Python dependencies
├── gsn2x               # gsn2x binary (Linux)
└── gsn2x-macOS         # gsn2x binary (macOS)
```

### Key classes

| Class | File | Role |
|-------|------|------|
| `BBN` | `bbn.py` | Core BBN wrapper — node/edge creation, inference, GSN YAML export, PNG render |
| `App` | `gui.py` | customtkinter main window — renders sliders, table, bar chart, assurance case image |
| `GoalNode` | `doe.py` | Root requirement node |
| `SuccessNode` | `doe.py` | Leaf node with a probability of success (tunable) |
| `MinThresholdNode` / `MaxThresholdNode` | `doe.py` | Intermediate threshold nodes with configurable pass/fail boundaries |

---

## Prerequisites

### System dependencies

```bash
# macOS
brew install cairo python-tk

# Ubuntu 20.04
sudo apt-get install libcairo2-dev python3-tk
```

### `gsn2x` binary

Pre-built binaries are included in the repo (`gsn2x` for Linux, `gsn2x-macOS` for macOS).

For other platforms, build from source:
```bash
git clone https://github.com/jonasthewolf/gsn2x
cd gsn2x
cargo build --release
```

Refer to [this discussion thread](https://github.com/jonasthewolf/gsn2x/discussions/333) for Ubuntu 20.04 specifics.

---

## Installation

```bash
git clone git@github.com:vishalgattani/pybbn.git
cd pybbn
pip install -r requirements.txt
```

### Key Python dependencies

| Package | Purpose |
|---------|---------|
| `pybbn` | Bayesian Belief Network inference (PPTC algorithm) |
| `customtkinter` | Modern tkinter GUI |
| `cairosvg` | SVG → PNG conversion for assurance case diagrams |
| `graphviz` | BBN graph visualization |
| `networkx` | Graph structure utilities |
| `matplotlib` | Bar chart plots embedded in GUI |
| `screeninfo` | Multi-monitor screen size detection |

---

## Usage

### Run the GUI

```bash
python gui.py
```

This will:
1. Build the mission BBN from `assurance_case.py`
2. Generate `assurance_case.yaml` and `assurance_case.png` (GSN diagram)
3. Launch the interactive tkinter window

### GUI controls

- **Sliders (right panel)** — tune input probabilities (`P(navigable terrain)`, `P(no collision)`, `P(pose within region)`) and thresholds; BBN updates live
- **Table** — shows posterior `True`/`False` probabilities for each requirement node
- **Bar charts** — visual breakdown of belief per requirement
- **Show Assurance Case** — displays the GSN PNG
- **Show Beliefs** — prints current posterior probabilities
- **Save Data** — (stub) export current state

### Define your own mission

Edit `assurance_case.py` to add/remove nodes and set initial probabilities:

```python
n_experiments = 5
p_correct_navigation = 0.9
p_no_collision = 0.1
p_correct_pose = 0.9
```

Add new nodes in `sample_mission_bbn()` using `GoalNode`, `SuccessNode`, `MinThresholdNode`, or `MaxThresholdNode`.

---

## Output files (generated at runtime)

| File | Description |
|------|-------------|
| `assurance_case.yaml` | GSN-formatted YAML of the BBN structure |
| `assurance_case.svg` | GSN diagram in SVG (intermediate) |
| `assurance_case.png` | GSN diagram in PNG (shown in GUI) |

---

## Development

```bash
# Linting + formatting
pre-commit install
pre-commit run --all-files

# Tests
python test.py
```

Pre-commit hooks: `black` (formatter), `isort` (import sorter), `flake8` (linter).

---

## Future Improvements

Numbered by priority — pick item 1 to start next:

1. **CI/CD pipeline** — add GitHub Actions workflow (lint + test on every push)
2. **Save/load BBN state** — export/import current slider state to JSON so sessions are resumable
3. **Config-driven missions** — define missions via YAML config instead of editing `assurance_case.py`
4. **Export belief history** — record how posteriors change as sliders are adjusted; export to CSV
5. **More node types** — AND/OR logic gates, weighted evidence nodes
6. **Web GUI** — Streamlit or Dash alternative to tkinter for browser-based access

---

## References

- [`pybbn` library](https://py-bbn.readthedocs.io/index.html) — Python BBN inference
- [`gsn2x`](https://github.com/jonasthewolf/gsn2x) — Goal Structuring Notation renderer
- [Goal Structuring Notation (GSN)](https://scsc.uk/gsn) — safety assurance standard
