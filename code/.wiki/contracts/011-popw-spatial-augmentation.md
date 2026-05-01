### CONTRACT #11: Add Spatial Augmentation (flip/crop) to PopW and IndustReal

WHAT:
  Implement spatial augmentation with horizontal flip and random crop for both PopW and IndustReal datasets, applied during training only.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/ikea_dataset.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/industreal_dataset.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/ikea_dataset.py
         /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/industreal_dataset.py
  RUN:   python -c "import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved'); from ikea_dataset import apply_spatial_aug; print('apply_spatial_aug imported')"

DONE_WHEN:
  - apply_spatial_aug(image: Tensor, boxes: Tensor, keypoints: Tensor) -> augmented tuple
  - Horizontal flip with probability 0.5 during training
  - Random crop: scale=[0.8, 1.0], aspect_ratio=[0.9, 1.1]
  - Keypoint flip indices: KEYPOINT_FLIP_PAIRS from config applied after horizontal flip
  - Box coordinates adjusted after crop/flip
  - No augmentation during validation/evaluation
  - Config: USE_SPATIAL_AUG = True

PROOF_FORMAT:
  CODE: `python -c "
import torch
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved')
from ikea_dataset import apply_spatial_aug
img = torch.rand(3, 480, 640)
boxes = torch.tensor([[100, 100, 200, 200]])
keypoints = torch.rand(17, 2)
aug_img, aug_boxes, aug_kpts = apply_spatial_aug(img, boxes, keypoints)
print(f'Augmented image shape: {aug_img.shape}')
"`

BLOCKER_IF:
  - Crop produces invalid boxes (negative coordinates)
  - Keypoint flip mapping incorrect for COCO 17-keypoint format

DEPENDS_ON: 1, 5
