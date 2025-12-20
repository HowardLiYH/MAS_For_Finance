"""FastAPI server for PopAgent Dashboard.

Provides REST API endpoints and WebSocket support for:
- Listing and fetching experiment data
- Real-time iteration updates during live/paper trading
- Experiment configuration and control
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..services.experiment_logger import (
    ExperimentLogger,
    load_experiment,
    list_experiments,
)


# =============================================================================
# Pydantic Models
# =============================================================================

class ExperimentListItem(BaseModel):
    experiment_id: str
    start_time: Optional[str]
    config: Dict[str, Any]
    completed: bool
    total_iterations: Optional[int]


class ExperimentDetail(BaseModel):
    experiment_id: str
    metadata: Optional[Dict[str, Any]]
    iterations: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]


class IterationDetail(BaseModel):
    experiment_id: str
    iteration: int
    data: Dict[str, Any]


class LiveUpdate(BaseModel):
    event: str
    experiment_id: str
    iteration: int
    data: Dict[str, Any]


# =============================================================================
# Connection Manager for WebSocket
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for live updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.experiment_subscriptions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)

        if experiment_id:
            if experiment_id not in self.experiment_subscriptions:
                self.experiment_subscriptions[experiment_id] = []
            self.experiment_subscriptions[experiment_id].append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Remove from all subscriptions
        for exp_id in list(self.experiment_subscriptions.keys()):
            if websocket in self.experiment_subscriptions[exp_id]:
                self.experiment_subscriptions[exp_id].remove(websocket)
            if not self.experiment_subscriptions[exp_id]:
                del self.experiment_subscriptions[exp_id]

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

    async def send_to_experiment(self, experiment_id: str, message: Dict[str, Any]):
        """Send to clients subscribed to a specific experiment."""
        if experiment_id in self.experiment_subscriptions:
            for connection in self.experiment_subscriptions[experiment_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


# =============================================================================
# FastAPI App
# =============================================================================

def create_app(log_dir: str = "logs/experiments") -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="PopAgent API",
        description="API for PopAgent Multi-Agent Trading Dashboard",
        version="1.0.0",
    )

    # CORS middleware for dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Connection manager for WebSocket
    manager = ConnectionManager()

    # Store log directory path
    app.state.log_dir = Path(log_dir)
    app.state.manager = manager

    # ==========================================================================
    # REST Endpoints
    # ==========================================================================

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "service": "PopAgent API", "version": "1.0.0"}

    @app.get("/experiments", response_model=List[ExperimentListItem])
    async def get_experiments():
        """List all experiments."""
        experiments = list_experiments(app.state.log_dir)
        return experiments

    @app.get("/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str):
        """Get full experiment data including all iterations."""
        try:
            data = load_experiment(app.state.log_dir, experiment_id)
            return data
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    @app.get("/experiments/{experiment_id}/iterations/{iteration}")
    async def get_iteration(experiment_id: str, iteration: int):
        """Get a specific iteration from an experiment."""
        try:
            data = load_experiment(app.state.log_dir, experiment_id)

            for iter_data in data["iterations"]:
                if iter_data.get("iteration") == iteration:
                    return {"experiment_id": experiment_id, "iteration": iteration, "data": iter_data}

            raise HTTPException(
                status_code=404,
                detail=f"Iteration {iteration} not found in experiment {experiment_id}"
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    @app.get("/experiments/{experiment_id}/summary")
    async def get_experiment_summary(experiment_id: str):
        """Get experiment summary."""
        try:
            data = load_experiment(app.state.log_dir, experiment_id)
            return data.get("summary", {})
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    @app.get("/experiments/{experiment_id}/learning-progress")
    async def get_learning_progress(experiment_id: str):
        """Get learning progress metrics over iterations."""
        try:
            data = load_experiment(app.state.log_dir, experiment_id)
            iterations = data.get("iterations", [])

            progress = []
            for iter_data in iterations:
                progress.append({
                    "iteration": iter_data.get("iteration"),
                    "best_pnl": iter_data.get("best_pnl", 0),
                    "avg_pnl": iter_data.get("avg_pnl", 0),
                    "diversity": iter_data.get("diversity_metrics", {}),
                    "transfer": iter_data.get("knowledge_transfer") is not None,
                })

            return {"experiment_id": experiment_id, "progress": progress}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    @app.get("/experiments/{experiment_id}/method-popularity")
    async def get_method_popularity(experiment_id: str):
        """Get method usage statistics across iterations."""
        try:
            data = load_experiment(app.state.log_dir, experiment_id)
            iterations = data.get("iterations", [])

            # Aggregate method usage
            usage: Dict[str, Dict[str, int]] = {}

            for iter_data in iterations:
                for decision in iter_data.get("agent_decisions", []):
                    role = decision.get("role")
                    if role not in usage:
                        usage[role] = {}

                    for method in decision.get("methods_selected", []):
                        usage[role][method] = usage[role].get(method, 0) + 1

            # Convert to percentages
            popularity: Dict[str, Dict[str, float]] = {}
            for role, methods in usage.items():
                total = sum(methods.values())
                popularity[role] = {m: c / total for m, c in methods.items()} if total > 0 else {}

            return {"experiment_id": experiment_id, "popularity": popularity}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    # ==========================================================================
    # WebSocket Endpoints
    # ==========================================================================

    @app.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket):
        """WebSocket for real-time iteration updates."""
        await manager.connect(websocket)

        try:
            while True:
                # Wait for messages from client
                data = await websocket.receive_json()

                # Handle subscription requests
                if data.get("action") == "subscribe":
                    experiment_id = data.get("experiment_id")
                    if experiment_id:
                        if experiment_id not in manager.experiment_subscriptions:
                            manager.experiment_subscriptions[experiment_id] = []
                        manager.experiment_subscriptions[experiment_id].append(websocket)
                        await websocket.send_json({
                            "event": "subscribed",
                            "experiment_id": experiment_id,
                        })

                elif data.get("action") == "unsubscribe":
                    experiment_id = data.get("experiment_id")
                    if experiment_id and experiment_id in manager.experiment_subscriptions:
                        if websocket in manager.experiment_subscriptions[experiment_id]:
                            manager.experiment_subscriptions[experiment_id].remove(websocket)
                        await websocket.send_json({
                            "event": "unsubscribed",
                            "experiment_id": experiment_id,
                        })

        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.websocket("/ws/experiments/{experiment_id}")
    async def websocket_experiment(websocket: WebSocket, experiment_id: str):
        """WebSocket for a specific experiment's updates."""
        await manager.connect(websocket, experiment_id)

        try:
            # Send current state
            try:
                data = load_experiment(app.state.log_dir, experiment_id)
                await websocket.send_json({
                    "event": "state",
                    "experiment_id": experiment_id,
                    "data": data,
                })
            except FileNotFoundError:
                await websocket.send_json({
                    "event": "error",
                    "message": f"Experiment {experiment_id} not found",
                })

            # Keep connection alive and wait for client messages
            while True:
                await websocket.receive_text()

        except WebSocketDisconnect:
            manager.disconnect(websocket)

    return app


# Create default app instance
app = create_app()


# =============================================================================
# Utility: Broadcast iteration update
# =============================================================================

async def broadcast_iteration(
    manager: ConnectionManager,
    experiment_id: str,
    iteration: int,
    data: Dict[str, Any],
):
    """Broadcast a new iteration to subscribed clients."""
    message = {
        "event": "iteration",
        "experiment_id": experiment_id,
        "iteration": iteration,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.send_to_experiment(experiment_id, message)


# =============================================================================
# CLI entry point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, log_dir: str = "logs/experiments"):
    """Run the API server."""
    import uvicorn

    app = create_app(log_dir=log_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
