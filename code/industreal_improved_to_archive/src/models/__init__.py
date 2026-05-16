# Re-export POPWMultiTaskModel so `from model import POPWMultiTaskModel` works
# (models/model.py is added to path via smoke_test.py sys.path manipulation)
from models.model import POPWMultiTaskModel

__all__ = ['POPWMultiTaskModel']
