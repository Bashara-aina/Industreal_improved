### CONTRACT #4: Implement GCN for skeleton topology in PopW

WHAT:
  Add Graph Convolutional Network module operating on skeleton topology (17 COCO keypoints) to ActivityHead for POPW, enabling spatial relationship modeling between body joints.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/model.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/model.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved'); from model import GCNSkeletonModule; print('GCNSkeletonModule imported')"

DONE_WHEN:
  - GCNSkeletonModule class implemented using torch_geometric or manual sparse matmul
  - Adjacency matrix defined for COCO 17-keypoint skeleton (left_right symmetric edges, limb connections)
  - Edges: (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),(5,9),(6,10),(9,11),(10,12),(11,13),(12,14),(9,10),(11,12)
  - Module takes [B, K, C] keypoint features and outputs [B, K, C] refined features
  - Config flag `USE_GCN_SKELETON = True` enables GCN in ActivityHead
  - GCN refines keypoint features before PoseFiLM or before activity classification
  - GCN uses 2 graph conv layers with hidden_dim=256, output_dim=256
  - Laplacian normalization: A_norm = I + D^(-1/2) * A * D^(-1/2)

PROOF_FORMAT:
  CODE: `python -c "
import torch
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved')
from model import GCNSkeletonModule
B, K, C = 2, 17, 256
module = GCNSkeletonModule(in_channels=C, hidden_channels=256, out_channels=256)
x = torch.randn(B, K, C)
out = module(x)
print(f'GCN output shape: {out.shape}')
"` → torch.Size([2, 17, 256])

BLOCKER_IF:
  - torch_geometric not available and manual sparse implementation fails
  - Adjacency matrix causes gradient explosion
  - GCN output dimension mismatch with downstream modules

DEPENDS_ON: 1
