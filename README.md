# digital-twin[README.md](https://github.com/user-attachments/files/26228495/README.md)
# Distributed Digital Twin — Smart Building Energy Optimisation

> **Quick start:** `pip install -e ".[dev]"` then `python experiments/run_experiment.py`

Full documentation is in [`docs/README.md`](docs/README.md).

## Project layout

```
digital_twin/
├── src/digital_twin/         ← Python package (all source)
│   ├── models.py             ← Data schemas
│   ├── sensor_simulator.py   ← Noisy sensor streams
│   ├── environment_simulator.py ← RC thermal model
│   ├── state_estimator.py    ← Kalman / particle filter
│   ├── forecast_engine.py    ← Exp smoothing / ARIMA
│   ├── optimiser.py          ← CVXPY MPC
│   ├── agent_coordinator.py  ← BuildingAgent + coordinator
│   ├── data_store.py         ← SQLite time-series store
│   ├── metrics.py            ← Evaluation metrics + plots
│   └── api.py                ← FastAPI REST server
├── tests/                    ← pytest test suite (37+ tests)
├── experiments/              ← Run scripts + registry
├── design/                   ← Diagrams (Mermaid)
├── docs/                     ← README, API spec, requirements
├── data/                     ← Example data files
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── CHANGELOG.md
```

## One-minute demo

```bash
# Install
pip install -e ".[dev]"

# Run 24-hour simulation (2 buildings, ~1 second)
python experiments/run_experiment.py --n-hours 24 --seed 42

# Start REST API
uvicorn digital_twin.api:create_app --factory --port 8000

# Run all tests
pytest tests/ -v
```

## Key design decisions

| Choice | Rationale |
|--------|-----------|
| CVXPY + OSQP | Open-source, warm-start, ≤ 5 s solve for H=12 |
| Kalman filter | Linear, efficient, closed-form; particle filter available as alternative |
| Double exp smoothing | No external dependency; degrades gracefully without statsmodels |
| SQLite backend | Zero-dependency, portable, queryable via pandas |
| Rule-based fallback | MPC gracefully degrades if solver unavailable |
| FastAPI | Async, auto-docs, pydantic validation |

See [`docs/README.md`](docs/README.md) for the complete guide.
