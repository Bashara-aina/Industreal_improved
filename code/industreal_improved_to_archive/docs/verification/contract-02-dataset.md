### CONTRACT #2: Create IndustReal dataset loader

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/industreal/industreal_dataset.py` — complete multi-task dataset loader for IndustReal. Handles AR_labels.csv (activity), OD_labels.json (COCO detection), PSR_labels_raw.csv (multi-label component states), pose.csv (head pose), hands.csv (hand joints). Single RGB camera, frame-level indexing.

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/ikea_dataset.py (full source, reuse _extract_boxes_from_coco pattern)
  READ:  /home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/train.csv (headerless: recording_id,action_class_id,action_desc,start_frame,end_frame)
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal/industreal_dataset.py

  Class structure:
  ```python
  class IndustRealMultiTaskDataset(Dataset):
      def __init__(self, split='train', img_size=(1280,720), augment=False, seed=42)
      def __getitem__(self, idx) -> Dict[str, Any]:
          # Returns:
          #   'image': torch.Tensor (3, 720, 1280) uint8
          #   'gt_boxes': torch.Tensor (N, 4) xyxy
          #   'gt_classes': torch.Tensor (N,) int64
          #   'head_pose': torch.Tensor (9,) float32 [forward_x,y,z, pos_x,y,z, up_x,y,z]
          #   'hand_joints': torch.Tensor (52,) float32 (26 joints × 2 coords, flatten)
          #   'psr_labels': torch.Tensor (11,) float32 (binary per component)
          #   'action_label': torch.Tensor (1,) int64
          #   'metadata': dict with recording_id, frame_num, camera='rgb'
  ```

  Key implementation details:
  1. Load recordings from split CSV (train.csv/val.csv/test.csv headerless, 5 cols)
  2. For each recording, build frame-level index: recording_id/frame_num -> activity_id, psr_labels
  3. PSR labels: use PSR_labels_raw.csv sparse format. For each frame, interpolate binary state from nearest change points. If frame < first change, use first state. If frame > last change, use last state.
  4. Detection: parse OD_labels.json per recording (COCO format, reuse _extract_boxes_from_coco from ikea_dataset)
  5. Head pose: parse pose.csv per frame (9 floats: forward_vec3, position3, up_vec3)
  6. Hand joints: parse hands.csv per frame (52 coords = 26 joints × 2, flatten xyxy... format)
  7. Activity: from AR_labels.csv — for each frame, find which (if any) action spans it. If none, use 0 (background/NA).
  8. COCO cache: same pattern as ikea_dataset.py (_PROC_COCO_CACHE)
  9. Augment: horizontal flip on image + flip head pose x-coords (forward_x, pos_x, up_x) and hand joint x-coords
  10. Image loading: single rgb/ folder, resize to img_size (1280×720)
  11. get_sampler(): class-balanced WeightedRandomSampler on activity_ids
  12. collate_fn: simple single-camera batch collation (NOT the multi-camera flatten from IKEA)

  PSR interpolation algorithm:
  ```
  For each recording with PSR_labels_raw.csv:
    1. Parse into list of (frame_num, [11 binary states])
    2. Sort by frame_num
    3. For any query frame:
       - Find nearest change point <= frame (prev_state)
       - Find nearest change point > frame (next_state) or use prev_state
       - Return prev_state
  ```

DONE_WHEN:
  - /home/newadmin/swarm-bot/project/popw/working/code/industreal/industreal_dataset.py exists and >400 lines
  - IndustRealMultiTaskDataset class defined with __init__, __getitem__, __len__, get_sampler, clear_coco_cache
  - collate_fn function defined
  - Dataset can be instantiated and len() returns >0 for train/val splits
  - __getitem__ returns all required keys: image, gt_boxes, gt_classes, head_pose, hand_joints, psr_labels, action_label, metadata

PROOF_FORMAT:
  python3 -c "
import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal')
import config as C
from industreal_dataset import IndustRealMultiTaskDataset
ds = IndustRealMultiTaskDataset(split='train', img_size=C.IMG_SIZE, augment=False)
print(f'Train dataset size: {len(ds)}')
assert len(ds) > 0, 'Empty train dataset'
item = ds[0]
required_keys = ['image', 'gt_boxes', 'gt_classes', 'head_pose', 'hand_joints', 'psr_labels', 'action_label', 'metadata']
for k in required_keys:
    assert k in item, f'Missing key: {k}'
print(f'Image shape: {item[\"image\"].shape}')
print(f'Head pose shape: {item[\"head_pose\"].shape}')
print(f'PSR labels shape: {item[\"psr_labels\"].shape}')
print(f'Action label: {item[\"action_label\"].item()}')
print('industreal_dataset.py OK')
"

BLOCKER_IF:
  - Source ikea_dataset.py not readable
  - train.csv path doesn't exist or has wrong format
  - Dataset __getitem__ raises KeyError or ShapeError

DEPENDS_ON: 1 (config.py must exist first)
