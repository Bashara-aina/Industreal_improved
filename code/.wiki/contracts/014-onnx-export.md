### CONTRACT #14: Add ONNX Export to PopW and IndustReal

WHAT:
  Implement ONNX export functionality for both PopW and IndustReal models to enable deployment optimization and inference speedup.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/model.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/export_onnx.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/export_onnx.py
  RUN:   python -c "import torch; print(f'PyTorch: {torch.__version__}'); import onnx; print(f'ONNX: {onnx.__version__}')"

DONE_WHEN:
  - export_onnx.py with export_popw(model, save_path) function
  - export_onnx.py with export_industreal(model, save_path) function
  - Exports single image forward pass (batch=1, 3xH/W)
  - Input shape: [1, 3, 480, 640] for PopW, [1, 3, 720, 1280] for IndustReal
  - Dynamic axes: batch as dynamic axis
  - opset_version=14 for latest ONNX operators
  - Verify export with onnx.checker.check_model()
  - ONNX inference produces same output shape as PyTorch

PROOF_FORMAT:
  CODE: `python -c "
import torch
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved')
from model import MultiTaskIKEA
net = MultiTaskIKEA(pretrained=False)
x = torch.randn(1, 3, 480, 640)
torch.onnx.export(net, x, 'test.onnx', opset_version=14)
import onnx
m = onnx.load('test.onnx')
onnx.checker.check_model(m)
print('ONNX export successful')
"` → no error output

BLOCKER_IF:
  - ONNX export fails due to dynamic control flow
  - PyTorch version incompatible with onnx export

DEPENDS_ON: 2, 6
