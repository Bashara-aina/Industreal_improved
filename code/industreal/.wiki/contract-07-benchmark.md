### CONTRACT #7: Copy benchmark.py (architecture-agnostic)

WHAT:
  Copy benchmark.py from IKEA popw_main to IndustReal directory, updating the model import to use MultiTaskIndustReal and config import to use IndustReal config.

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/benchmark.py (full source)
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal/benchmark.py

  Changes:
  - Change `from model import MultiTaskIKEA` → `from model import MultiTaskIndustReal`
  - Change `MultiTaskIKEA(pretrained=..., use_film=C.USE_FILM)` → `MultiTaskIndustReal(pretrained=...)`
  - Remove `use_film` parameter from model instantiation (IndustReal has no FiLM)
  - Keep all benchmark functions identical (measure_gflops, measure_fps, measure_fps_batched)
  - Keep argparse, logging, and main() identical except model class name

DONE_WHEN:
  - /home/newadmin/swarm-bot/project/popw/working/code/industreal/benchmark.py exists
  - Benchmark runs without import errors
  - python benchmark.py produces GFLOPs, FPS, and memory metrics

PROOF_FORMAT:
  python3 -c "
import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal')
from benchmark import measure_gflops, measure_fps, measure_fps_batched
from model import MultiTaskIndustReal
import torch
model = MultiTaskIndustReal(pretrained=False)
print('benchmark.py imports OK')
print('Model loaded OK')
"

BLOCKER_IF:
  - benchmark.py fails to import due to model/config mismatch

DEPENDS_ON: 1, 3 (config.py and model.py must exist first)
