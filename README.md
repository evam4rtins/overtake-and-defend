# overtake-and-defend

Overtaking and defending in a race-track corner: optimization-based local planning vs. learned driving (reinforcement learning).
Course project for *Automation and Control in Autonomous Vehicles* (ACAV).

- **Phase 1:** the ego car uses the local planners of the course (sigmoid and lattice) to overtake an opponent that follows a fixed racing line.
- **Phase 2:** both cars learn: the attacker learns to execute fixed overtaking strategies, the defender learns freely to stay ahead.


## Requirements

- Python **3.10 or newer** (developed with 3.13)
- git

Check with `python3 --version`.

## Setup

Do this once, from the folder where you want the project:

```bash
git clone <repo-url> overtake-and-defend
cd overtake-and-defend

python3 -m venv .venv            # create the virtual environment
source .venv/bin/activate        # activate it (Windows: .venv\Scripts\activate)

pip install --upgrade pip
pip install -e ".[dev]"          # dependencies + the package itself + pytest

pytest                           # all tests should pass
```

For Phase 2 (reinforcement learning), add the extra dependencies:

```bash
pip install -e ".[rl]"           # gymnasium, stable-baselines3, tensorboard
```

## Editor (VS Code)

Select the interpreter of the virtual environment, otherwise imports are underlined in red:
**Cmd/Ctrl + Shift + P → "Python: Select Interpreter" → `.venv`**.


## Project layout

```
overtake-and-defend/
├── src/overtake/      the Python package 
├── tests/             one test file per module
├── configs/           scenarios in YAML
├── scripts/           command-line entry points 
└── pyproject.toml     package definition and dependenciesn
```

## Conventions

- **Units:** SI everywhere (m, s, rad, m/s, m/s²).
- **Frames:** global (x, y), heading ψ counter-clockwise from the x axis; Frenet (s, d) with d > 0 to the left of the centreline.
- **Shared types:** all modules exchange the dataclasses of `src/overtake/types.py` (`VehicleParams`, `VehicleState`, `Trajectory`, `CarView`, `Observation`).
