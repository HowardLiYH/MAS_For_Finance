# API module for PopAgent dashboard
"""FastAPI server for serving experiment data and live updates."""

from .server import create_app, app

__all__ = ["create_app", "app"]
