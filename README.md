# pybbn — Bayesian Belief Network Assurance Case Tool

A robotics safety assurance tool that uses **Bayesian Belief Networks (BBN)** to probabilistically verify whether a robot mission meets its requirements. Visualizes the assurance case as a **Goal Structuring Notation (GSN)** diagram via a modern **FastAPI + React** web interface.

---

## What It Does

Given a robot mission (e.g. navigate terrain, avoid collisions, maintain pose accuracy), this tool:

1. **Builds a BBN** — models each requirement as a probabilistic node with conditional probability tables (CPTs)
2. **Runs inference** — propagates probabilities through the network using the PPTC algorithm
3. **Generates a GSN assurance case** — exports the belief network as a YAML-based Goal Structuring Notation diagram (via `gsn2x`) rendered as an inline SVG
4. **Serves a web UI** — FastAPI backend exposes BBN computation as a REST API; React frontend provides interactive sliders, probability tables, bar charts, and live SVG rendering

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
├── src/
│   └── pybbn_assurance/
│       ├── __init__.py        # Package exports
│       ├── bbn.py             # BBN class — wraps pybbn library, inference, GSN export
│       ├── doe.py             # Node types (GoalNode, SuccessNode, ThresholdNode)
│       ├── helper.py          # Utility functions
│       ├── logger.py          # Logging setup
│       └── cases/
│           ├── __init__.py
│           └── mission.py     # Mission definition — builds the BBN
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app — CORS, router, static file serving
│   ├── models.py              # Pydantic request/response schemas
│   └── routers/
│       ├── __init__.py
│       └── bbn.py             # POST /api/bbn/compute endpoint
├── frontend/
│   ├── index.html             # Vite entry HTML
│   ├── package.json           # React + Vite + Recharts + Radix UI
│   ├── vite.config.js         # Vite config with /api proxy → localhost:8000
│   └── src/
│       ├── main.jsx           # React entry point
│       ├── App.jsx            # Layout: SliderPanel + AssuranceCaseSVG + ProbabilityTable + BeliefBarChart
│       └── components/
│           ├── SliderPanel.jsx         # 6 sliders with debounced API calls
│           ├── AssuranceCaseSVG.jsx    # Inline SVG rendering with zoom controls
│           ├── ProbabilityTable.jsx    # Node probability table with color coding
│           └── BeliefBarChart.jsx      # Recharts bar chart (True/False per node)
├── tests/
│   ├── __init__.py
│   ├── test_api.py           # FastAPI endpoint tests (httpx TestClient)
│   ├── test_bbn.py           # BBN computation tests
│   ├── test_doe.py           # Node type tests
│   ├── test_helper.py        # Helper function tests
│   ├── test_logger.py        # Logger tests
│   └── test_mission.py       # Mission definition tests
├── pyproject.toml             # Project config, dependencies, optional deps
├── .gitignore
└── README.md                  # This file
```

### Data flow

```
Browser (React)
    │
    │  POST /api/bbn/compute  { BBNParams }
    ▼
FastAPI (api/routers/bbn.py)
    │
    │  builds BBN, runs inference, reads SVG
    ▼
BBN Engine (src/pybbn_assurance/bbn.py)
    │
    │  pybbn library (PPTC algorithm)
    │  gsn2x binary (GSN YAML → SVG)
    ▼
Response: { nodes: [...], assurance_case_svg: "<svg>...</svg>" }
    │
    ▼
Browser renders: SVG diagram + probability table + bar chart
```

### Key classes

| Class | File | Role |
|-------|------|------|
| `BBN` | `src/pybbn_assurance/bbn.py` | Core BBN wrapper — node/edge creation, inference, GSN YAML export, SVG render |
| `GoalNode` | `src/pybbn_assurance/doe.py` | Root requirement node |
| `SuccessNode` | `src/pybbn_assurance/doe.py` | Leaf node with a probability of success (tunable) |
| `MinThresholdNode` / `MaxThresholdNode` | `src/pybbn_assurance/doe.py` | Intermediate threshold nodes with configurable pass/fail boundaries |

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** and npm
- **`gsn2x` binary** — pre-built binaries included (`gsn2x` for Linux, `gsn2x-macOS` for macOS). For other platforms: `cargo build --release` from [jonasthewolf/gsn2x](https://github.com/jonasthewolf/gsn2x)

---

## Installation

```bash
git clone git@github.com:vishalgattani/pybbn.git
cd pybbn
pip install -e ".[api]"
cd frontend && npm install
```

### Python dependencies

| Package | Purpose |
|---------|---------|
| `pybbn` | Bayesian Belief Network inference (PPTC algorithm) |
| `fastapi` | Web framework for BBN API |
| `uvicorn[standard]` | ASGI server |
| `httpx` | HTTP client (for tests) |
| `pydantic` | Request/response validation |
| `graphviz` | BBN graph visualization |
| `networkx` | Graph structure utilities |
| `matplotlib` | Plotting utilities |
| `pyyaml` | YAML parsing for GSN export |

### Frontend dependencies

| Package | Purpose |
|---------|---------|
| `react` / `react-dom` | UI framework |
| `recharts` | Bar chart component |
| `@radix-ui/react-slider` | Accessible slider component |
| `clsx` | Conditional CSS classes |
| `vite` | Build tool / dev server |

---

## Running Locally

Two terminals required:

```bash
# Terminal 1 — Backend
cd ~/Desktop/my-brain-in-logseq/projects/github/pybbn
pip install -e ".[api]"
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The Vite dev server proxies `/api` requests to `http://localhost:8000`, so the frontend can call the backend seamlessly.

### Production build

```bash
# Build frontend
cd frontend && npm run build

# Run backend (serves frontend/dist as static files)
uvicorn api.main:app --port 8000
# → http://localhost:8000
```

---

## API Documentation

### `POST /api/bbn/compute`

Build a BBN, run inference, and return node probabilities and the assurance case SVG.

**Request body** (`BBNParams`):

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `p_correct_navigation` | `float` | `0.9` | `0.0–1.0` | P(robot on navigable terrain) |
| `p_no_collision` | `float` | `0.1` | `0.0–1.0` | P(robot does not collide) |
| `p_correct_pose` | `float` | `0.9` | `0.0–1.0` | P(robot pose within region) |
| `nav_threshold` | `int` | `0` | `0–10000` | Navigation threshold |
| `collision_threshold` | `int` | `0` | `0–10000` | Collision threshold |
| `pose_threshold` | `int` | `0` | `0–10000` | Pose threshold |

All fields are optional; omit any to use its default value.

**Response** (`BBNResult`):

```json
{
  "nodes": [
    { "name": "Robot Nav Terrain under Threshold", "p_true": 0.85, "p_false": 0.15 },
    { "name": "Robot Collision under Threshold", "p_true": 0.72, "p_false": 0.28 },
    { "name": "Robot Pose under Threshold", "p_true": 0.85, "p_false": 0.15 },
    { "name": "Meeting Requirements", "p_true": 0.55, "p_false": 0.45 }
  ],
  "assurance_case_svg": "<svg xmlns=\"...\">...</svg>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `[{ name: string, p_true: float, p_false: float }]` | Posterior probability for each BBN node |
| `assurance_case_svg` | `string` | Inline SVG of the GSN assurance case diagram |

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/pybbn_assurance --cov=api --cov-report=term-missing

# Run specific test file
pytest tests/test_api.py -v
```

### Test coverage target: ≥ 80%

### Test files

| File | What it tests | Count |
|------|--------------|-------|
| `test_api.py` | FastAPI endpoints — default params, custom params, invalid input validation | 7 tests |
| `test_bbn.py` | BBN construction, inference, SVG generation | — |
| `test_doe.py` | Node type creation and behavior | — |
| `test_helper.py` | Utility functions | — |
| `test_logger.py` | Logging configuration | — |
| `test_mission.py` | Mission definition and BBN integration | — |

91 existing BBN/DOE tests + 7 API tests.

---

## Development

```bash
# Linting + formatting
pre-commit install
pre-commit run --all-files
```

Pre-commit hooks: `black` (formatter), `isort` (import sorter), `flake8` (linter).

---

## What changed from Phase 1 (feat/revamp)

This branch (`feat/web-stack`) replaces the old `customtkinter` GUI with a FastAPI + React web stack:

- **Removed:** `gui.py`, `widgets/` directory (`sliders.py`, `table.py`), `ctksliders.py`, `ctktable.py`, `custom_sliders.py`
- **Removed dependencies:** `customtkinter`, `darkdetect`, `screeninfo`, `cairosvg`
- **Added:** `api/` package (FastAPI backend with `/api/bbn/compute` endpoint), `frontend/` package (React + Vite with SliderPanel, ProbabilityTable, AssuranceCaseSVG, BeliefBarChart), `get_assurance_case_svg()` method on `BBN` class
- **Unchanged:** `src/pybbn_assurance/` BBN/DOE computation logic — all existing tests still pass (≥80% coverage)

---

## Future Improvements

Numbered by priority — pick item 1 to start next:

1. **CI/CD pipeline** — add GitHub Actions workflow (lint + test on every push)
2. **Save/load BBN state** — export/import current slider state to JSON so sessions are resumable
3. **Config-driven missions** — define missions via YAML config instead of editing `cases/mission.py`
4. **Export belief history** — record how posteriors change as sliders are adjusted; export to CSV
5. **More node types** — AND/OR logic gates, weighted evidence nodes
6. **Authentication** — API key or OAuth for production deployment
7. **Docker support** — containerized deployment for the full stack

---

## References

- [`pybbn` library](https://py-bbn.readthedocs.io/index.html) — Python BBN inference
- [`gsn2x`](https://github.com/jonasthewolf/gsn2x) — Goal Structuring Notation renderer
- [Goal Structuring Notation (GSN)](https://scsc.uk/gsn) — safety assurance standard
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework
- [React](https://react.dev/) — UI framework
- [Vite](https://vitejs.dev/) — Frontend build tool
