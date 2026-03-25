"""
data_store.py
-------------
In-memory time-series store with optional SQLite persistence.

The store supports:
  - Appending TelemetryPoint, AgentState, OptimisationResult records.
  - Querying by agent_id, time range, and sensor type.
  - Exporting to pandas DataFrame and CSV.
  - SQLite backend (opt-in) for persistence across runs.

Public API
----------
DataStore  – primary interface
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

from .models import AgentState, OptimisationResult, TelemetryPoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory + SQLite DataStore
# ---------------------------------------------------------------------------

class DataStore:
    """Unified store for telemetry, agent states, and optimisation results.

    Parameters
    ----------
    db_path : Path to SQLite file.  Pass ':memory:' (default) for pure
              in-memory operation.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)
        self._create_schema()
        log.debug(f"DataStore initialised (db_path={db_path!r})")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        """Create all tables if they do not exist."""
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL    NOT NULL,
                    sensor_id   TEXT    NOT NULL,
                    sensor_type TEXT    NOT NULL,
                    value       REAL    NOT NULL,
                    unit        TEXT,
                    quality     REAL    DEFAULT 1.0,
                    agent_id    TEXT    NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tel_agent_time ON telemetry(agent_id, timestamp)")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL    NOT NULL,
                    agent_id    TEXT    NOT NULL,
                    payload     TEXT    NOT NULL   -- JSON blob
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_as_agent_time ON agent_states(agent_id, timestamp)")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS opt_results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       REAL    NOT NULL,
                    agent_id        TEXT    NOT NULL,
                    status          TEXT,
                    solve_time_s    REAL,
                    objective_value REAL,
                    payload         TEXT    NOT NULL   -- JSON blob
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_or_agent_time ON opt_results(agent_id, timestamp)")

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def append_telemetry(self, points: List[TelemetryPoint]) -> None:
        """Insert a batch of TelemetryPoint records."""
        with self._cursor() as cur:
            cur.executemany(
                """INSERT INTO telemetry
                   (timestamp, sensor_id, sensor_type, value, unit, quality, agent_id)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (p.timestamp, p.sensor_id, p.sensor_type.value,
                     p.value, p.unit, p.quality, p.agent_id)
                    for p in points
                ],
            )

    def append_agent_state(self, state: AgentState) -> None:
        """Insert one AgentState snapshot."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO agent_states (timestamp, agent_id, payload) VALUES (?,?,?)",
                (state.timestamp, state.agent_id, json.dumps(state.to_dict())),
            )

    def append_opt_result(self, result: OptimisationResult) -> None:
        """Insert one OptimisationResult."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO opt_results
                   (timestamp, agent_id, status, solve_time_s, objective_value, payload)
                   VALUES (?,?,?,?,?,?)""",
                (
                    result.timestamp,
                    result.agent_id,
                    result.status,
                    result.solve_time_s,
                    result.objective_value,
                    json.dumps(result.to_dict()),
                ),
            )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_telemetry(
        self,
        agent_id:    Optional[str]   = None,
        sensor_type: Optional[str]   = None,
        t_start:     Optional[float] = None,
        t_end:       Optional[float] = None,
    ) -> pd.DataFrame:
        """Query telemetry with optional filters.

        Returns
        -------
        DataFrame with columns: timestamp, sensor_id, sensor_type, value,
                                unit, quality, agent_id.
        """
        sql    = "SELECT timestamp, sensor_id, sensor_type, value, unit, quality, agent_id FROM telemetry WHERE 1=1"
        params: List[Any] = []
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if sensor_type:
            sql += " AND sensor_type = ?"
            params.append(sensor_type)
        if t_start is not None:
            sql += " AND timestamp >= ?"
            params.append(t_start)
        if t_end is not None:
            sql += " AND timestamp <= ?"
            params.append(t_end)
        sql += " ORDER BY timestamp ASC"
        return pd.read_sql_query(sql, self._conn, params=params)

    def get_opt_results(
        self,
        agent_id: Optional[str]   = None,
        t_start:  Optional[float] = None,
        t_end:    Optional[float] = None,
    ) -> pd.DataFrame:
        """Query optimisation results table."""
        sql    = "SELECT timestamp, agent_id, status, solve_time_s, objective_value FROM opt_results WHERE 1=1"
        params: List[Any] = []
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if t_start is not None:
            sql += " AND timestamp >= ?"
            params.append(t_start)
        if t_end is not None:
            sql += " AND timestamp <= ?"
            params.append(t_end)
        sql += " ORDER BY timestamp ASC"
        return pd.read_sql_query(sql, self._conn, params=params)

    def latest_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent AgentState dict for an agent."""
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT payload FROM agent_states WHERE agent_id=? ORDER BY timestamp DESC LIMIT 1",
                (agent_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_csv(self, path: str) -> None:
        """Export telemetry table to CSV."""
        df = self.get_telemetry()
        df.to_csv(path, index=False)
        log.info(f"Telemetry exported to {path!r} ({len(df)} rows)")

    def export_metrics_csv(self, path: str) -> None:
        """Export aggregated metrics per timestep to CSV."""
        tel_df = self.get_telemetry()
        opt_df = self.get_opt_results()

        if tel_df.empty:
            log.warning("No telemetry data to export.")
            return

        # Pivot power readings
        power_df = tel_df[tel_df.sensor_type == "power"].pivot_table(
            index="timestamp", columns="sensor_id", values="value", aggfunc="mean"
        ).reset_index()

        # Merge with solve times
        if not opt_df.empty:
            merged = power_df.merge(
                opt_df[["timestamp", "agent_id", "solve_time_s", "objective_value", "status"]],
                on="timestamp",
                how="left",
            )
        else:
            merged = power_df

        merged.to_csv(path, index=False)
        log.info(f"Metrics exported to {path!r} ({len(merged)} rows)")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all rows from all tables (useful between experiments)."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM telemetry")
            cur.execute("DELETE FROM agent_states")
            cur.execute("DELETE FROM opt_results")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()
