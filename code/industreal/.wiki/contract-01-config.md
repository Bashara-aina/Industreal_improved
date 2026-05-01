### CONTRACT #1: Create IndustReal config

WHAT:
  Create `/home/newadmin/swarm-bot/project/popw/working/code/industreal/config.py` by adapting the IKEA POPW config.py for the IndustReal dataset. Single camera, 24 ASD classes, 74 AR classes, 36 PSR steps, 9-DoF head pose, 11-component PSR.

FILES:
  READ:  /media/newadmin/master/POPW/popw_main/config.py (full source)
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal/config.py

  Key changes from IKEA config:
  - POPW_ROOT → IndustReal dataset root
  - OUTPUT_ROOT → .../working/code/industreal/runs
  - Remove IKEA_RAW_ROOT, COCO_RAW_ROOT, ANNO_RAW_ROOT, GITHUB_ROOT
  - FURNITURE_TYPES → recording patterns (e.g., all recordings indexed by split CSVs)
  - CAMERAS = ['rgb'] (single camera, 1280×720)
  - NUM_DET_CLASSES = 24 (assembly state categories from OD_labels.json)
  - DET_CLASS_NAMES = {1:'background', 2:'10000000000', 3:'10010010000', ..., 24:'error_state'}
  - NUM_CLASSES_ACT = 74 (non-consecutive IDs 0-74 from AR_labels.csv)
  - ACT_CLASS_NAMES: load from AR_labels.csv unique action_class_ids, fill gaps with f'action_{id}'
  - NUM_PSR_STEPS = 36 (procedure step IDs 0-35)
  - PSR_NUM_COMPONENTS = 11 (from PSR_labels_raw.csv)
  - NUM_KEYPOINTS = 9 (head pose: forward_vec3 + position3 + up_vec3)
  - HEAD_POSE_NAMES = ['forward_x', 'forward_y', 'forward_z', 'pos_x', 'pos_y', 'pos_z', 'up_x', 'up_y', 'up_z']
  - IMG_WIDTH = 1280, IMG_HEIGHT = 720 (from actual IndustReal frames)
  - Keep IMAGENET_MEAN/STD, BATCH_SIZE=8, BASE_LR=1e-4, EPOCHS=150
  - Remove multi-camera paths, DATASET_MODE, USE_FILM, PRESETS system
  - Remove VAL_RATIO, TRAIN_SPLIT_FILE (use existing train.csv/val.csv/test.csv)
  - TRAIN_DET=True, TRAIN_POSE_HEAD=True, TRAIN_ACT=True, TRAIN_PSR=True (new task)
  - PSR_LOSS_TYPE='bce' (binary cross-entropy per component)

DONE_WHEN:
  - /home/newadmin/swarm-bot/project/popw/working/code/industreal/config.py exists and >200 lines
  - DET_CLASS_NAMES has 24 entries (IDs 1-24)
  - ACT_CLASS_NAMES has 74 entries covering all AR class IDs
  - CAMERAS = ['rgb']
  - IMG_WIDTH=1280, IMG_HEIGHT=720

PROOF_FORMAT:
  python3 -c "
import sys; sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal')
import config as C
assert C.NUM_DET_CLASSES == 24, f'Expected 24, got {C.NUM_DET_CLASSES}'
assert C.NUM_CLASSES_ACT == 74, f'Expected 74, got {C.NUM_CLASSES_ACT}'
assert C.NUM_PSR_STEPS == 36, f'Expected 36, got {C.NUM_PSR_STEPS}'
assert C.PSR_NUM_COMPONENTS == 11, f'Expected 11, got {C.PSR_NUM_COMPONENTS}'
assert C.NUM_KEYPOINTS == 9, f'Expected 9, got {C.NUM_KEYPOINTS}'
assert C.CAMERAS == ['rgb'], f'Expected [\"rgb\"], got {C.CAMERAS}'
assert C.IMG_WIDTH == 1280 and C.IMG_HEIGHT == 720
assert hasattr(C, 'DET_CLASS_NAMES') and len(C.DET_CLASS_NAMES) == 24
print('config.py OK')
"

BLOCKER_IF:
  - Source config.py not readable (symlink outside allowed dirs)
  - Any assertion fails (wrong class counts, missing attributes)

DEPENDS_ON: none
