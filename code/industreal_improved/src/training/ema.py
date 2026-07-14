"""
EMA — re-export from src.models.model for backward compatibility.
The canonical EMA class lives in src.models.model.ModelEMA.
"""

from src.models.model import EMA as ModelEMA

__all__ = ["ModelEMA"]
