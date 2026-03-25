"""
api.py
------
FastAPI REST API for controlling and monitoring the digital twin.

Endpoints
---------
GET  /health                  – health check
GET  /agents                  – list registered agents
GET  /agents/{id}/state       – latest agent state
GET  /agents/{id}/telemetry   – recent telemetry readings
POST /agents/{id}/step        – advance one control cycle
GET  /metrics                 – summary metrics
POST /experiment/start        – start a batch simulation run
GET  /experiment/status       – poll experiment progress
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .agent_coordinator import AgentConfig, AgentCoordinator, BuildingAgent, CoordinatorConfig
from .data_store import DataStore
from .metrics import compute_metrics

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    agents: Optional[List[BuildingAgent]] = None,
    coordinator: Optional[AgentCoordinator] = None,
    store: Optional[DataStore] = None,
) -> Any:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    agents      : Pre-built agent list (or None to use defaults).
    coordinator : Pre-built coordinator (or None to use default).
    store       : DataStore instance (or None for in-memory default).

    Returns
    -------
    FastAPI app instance.
    """
    if not HAS_FASTAPI:
        raise ImportError("fastapi and pydantic are required to run the API server.")

    app = FastAPI(
        title="Distributed Digital Twin API",
        description="Smart Building Energy Optimisation — REST interface",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Defaults
    _agents = agents or [
        BuildingAgent(AgentConfig("building_a", seed=42)),
        BuildingAgent(AgentConfig("building_b", seed=43)),
    ]
    _coordinator = coordinator or AgentCoordinator(_agents, CoordinatorConfig())
    _store       = store or DataStore()

    # Experiment state
    _experiment = {"running": False, "progress": 0, "total": 0, "results": None}
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    # Request/response models
    # ------------------------------------------------------------------

    class StepRequest(BaseModel):
        unix_time:    float
        outdoor_temp: float = 10.0
        day_of_year:  int   = 180

    class ExperimentRequest(BaseModel):
        n_hours:      int   = 24
        outdoor_mean: float = 10.0
        seed:         int   = 42

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok", "timestamp": time.time()}

    @app.get("/agents")
    def list_agents():
        return {"agents": [a.agent_id for a in _agents]}

    @app.get("/agents/{agent_id}/state")
    def get_agent_state(agent_id: str):
        agent = next((a for a in _agents if a.agent_id == agent_id), None)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
        if agent.current_state is None:
            raise HTTPException(status_code=204, detail="No state yet — run a step first.")
        return agent.current_state.to_dict()

    @app.get("/agents/{agent_id}/telemetry")
    def get_telemetry(agent_id: str, limit: int = 100):
        df = _store.get_telemetry(agent_id=agent_id)
        if df.empty:
            return {"rows": []}
        return {"rows": df.tail(limit).to_dict(orient="records")}

    @app.post("/agents/{agent_id}/step")
    def step_agent(agent_id: str, req: StepRequest):
        agent = next((a for a in _agents if a.agent_id == agent_id), None)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
        result = agent.control_cycle(
            unix_time=req.unix_time,
            outdoor_temp=req.outdoor_temp,
        )
        if agent.current_state:
            _store.append_agent_state(agent.current_state)
        _store.append_opt_result(result)
        return result.to_dict()

    @app.post("/coordinator/step")
    def coordinator_step(req: StepRequest):
        results = _coordinator.step(
            unix_time=req.unix_time,
            outdoor_temp=req.outdoor_temp,
            day_of_year=req.day_of_year,
        )
        return {aid: r.to_dict() for aid, r in results.items()}

    @app.get("/metrics")
    def get_metrics(agent_id: Optional[str] = None):
        target_agents = [a for a in _agents if agent_id is None or a.agent_id == agent_id]
        if not target_agents:
            raise HTTPException(status_code=404, detail="No matching agents.")
        all_history = []
        for a in target_agents:
            all_history.extend(a.history)
        all_history.sort(key=lambda x: x["timestamp"])
        return compute_metrics(all_history)

    @app.post("/experiment/start")
    def start_experiment(req: ExperimentRequest):
        with _lock:
            if _experiment["running"]:
                raise HTTPException(status_code=409, detail="Experiment already running.")
            _experiment["running"]  = True
            _experiment["progress"] = 0

        def _run():
            from .environment_simulator import EnvironmentSimulator
            import numpy as np
            rng = np.random.default_rng(req.seed)
            dt  = 300.0
            n_steps = req.n_hours * 12
            _experiment["total"] = n_steps
            t0 = 0.0

            for step in range(n_steps):
                t = t0 + step * dt
                T_out = req.outdoor_mean + 5.0 * np.sin(2 * np.pi * (t % 86400) / 86400 - np.pi / 2)
                _coordinator.step(unix_time=t, outdoor_temp=float(T_out))
                with _lock:
                    _experiment["progress"] = step + 1

            # Aggregate results
            all_history = []
            for a in _agents:
                all_history.extend(a.history)
            with _lock:
                _experiment["results"] = compute_metrics(all_history)
                _experiment["running"] = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"message": "Experiment started.", "n_steps": req.n_hours * 12}

    @app.get("/experiment/status")
    def experiment_status():
        with _lock:
            return {
                "running":  _experiment["running"],
                "progress": _experiment["progress"],
                "total":    _experiment["total"],
                "results":  _experiment["results"],
            }

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:  # pragma: no cover
    """Launch the Uvicorn server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")
    app = create_app()
    uvicorn.run(app, host=host, port=port)
