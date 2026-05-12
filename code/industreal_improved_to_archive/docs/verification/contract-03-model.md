### CONTRACT #3: Create IndustReal multi-task model

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/industreal/model.py` — MultiTaskIndustReal with ResNet50-FPN backbone and 4 task heads: DetectionHead (ASD, 24 classes), HeadPoseHead (9-DoF regression), ActivityHead (74 classes), PSRHead (11-component multi-label).

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/model.py (full source, reuse DetectionHead, ActivityHead, FPN, AnchorGenerator)
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal/model.py

  Architecture:
  ```
  ResNet50 (ImageNet pretrained)
    └── FPN (256ch)
          ├── DetectionHead (RetinaNet-style, 24 classes)
          ├── HeadPoseHead (regress 9-DoF head pose from C5)
          ├── ActivityHead (GAP C5+P4 → FC, 74 classes)
          └── PSRHead (GAP C5 → FC, 11 binary outputs)
  ```

  HeadPoseHead: single-camera, no heatmaps. C5 GAP → FC(512) → FC(9). Use L1 loss for training. No soft-argmax needed. Direct regression of [forward_vec3, pos_vec3, up_vec3] normalized to [-1,1] range using training data statistics.

  PSRHead: C5 GAP → FC(256) → FC(11). Sigmoid activation for multi-label binary classification. BCEWithLogitsLoss during training.

  ActivityHead: identical architecture to IKEA version (C5+P4 fusion, residual, dropout=0.3), but num_classes=74.

  DetectionHead: identical RetinaNet-style to IKEA (9 anchors, 4 conv layers each for cls/reg subnets), but num_classes=24.

  Forward pass returns dict with:
    - cls_preds, reg_preds, anchors (for detection)
    - head_pose (9-DoF tensor)  
    - act_logits (74 classes)
    - psr_logits (11 binary)

  Remove PoseFiLMModule, PoseCrossAttentionModule, multi-camera complexity.

  count_parameters() function must work for MultiTaskIndustReal.

DONE_WHEN:
  - /home/newadmin/swarm-bot/project/popw/working/code/industreal/model.py exists and >400 lines
  - MultiTaskIndustReal class defined with forward() returning all 5 required tensors
  - DetectionHead, ActivityHead reused from IKEA pattern (same architecture, different num_classes)
  - HeadPoseHead regresses 9-DoF from C5 GAP
  - PSRHead outputs 11 binary logits with sigmoid
  - count_parameters() works
  - Model can be instantiated with pretrained=True and forward pass succeeds

PROOF_FORMAT:
  python3 -c "
import sys, torch
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal')
import config as C
from model import MultiTaskIndustReal, count_parameters

model = MultiTaskIndustReal(pretrained=True).train()
images = torch.randn(2, 3, C.IMG_HEIGHT, C.IMG_WIDTH)
outputs = model(images)

required_keys = ['cls_preds', 'reg_preds', 'anchors', 'head_pose', 'act_logits', 'psr_logits']
for k in required_keys:
    assert k in outputs, f'Missing output key: {k}'
    assert outputs[k] is not None, f'None output: {k}'

print(f'cls_preds shape: {outputs[\"cls_preds\"].shape}')
print(f'head_pose shape: {outputs[\"head_pose\"].shape}')
print(f'act_logits shape: {outputs[\"act_logits\"].shape}')
print(f'psr_logits shape: {outputs[\"psr_logits\"].shape}')

params = count_parameters(model)
print(f'Total params: {params[\"total_all\"]:,}')
print('model.py OK')
"

BLOCKER_IF:
  - Source model.py not readable
  - Model forward fails with shape mismatch
  - Any required output key missing

DEPENDS_ON: 1 (config.py must exist first)
