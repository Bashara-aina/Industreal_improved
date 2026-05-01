### CONTRACT #2: Implement ConvNeXt-Tiny backbone for PopW

WHAT:
  Replace ResNet-50 backbone with ConvNeXt-Tiny in popw_main_improved/model.py, maintaining FPN compatibility and frozen BN strategy.

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/model.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved/model.py
  RUN:   python -c "import timm; m = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=False); print(sum(p.numel() for p in m.parameters()))"

DONE_WHEN:
  - ConvNeXtBackbone class replaces ResNet50 in model.py
  - ConvNeXt-Tiny outputs C3(192ch), C4(384ch), C5(768ch) matching FPN [192, 384, 768] input
  - Backbone uses `fb_in22k_ft_in1k` pretrained (ImageNet-22k finetuned to ImageNet-1k)
  - BN layers frozen via `m.eval(); for p in m.parameters(): p.requires_grad=False`
  - Total trainable params < 28M (backbone alone)
  - Model forward pass works with same interface as original (returns dict with cls_preds, reg_preds, etc.)

PROOF_FORMAT:
  CODE: `cd /home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved && python -c "
import torch, model
net = model.MultiTaskIKEA(pretrained=False)
params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Trainable params: {params:,}')
x = torch.randn(1, 3, 480, 640)
out = net(x)
print(f'Output keys: {list(out.keys())}')
"` → should print params count and output keys

BLOCKER_IF:
  - FPN lateral connections fail due to channel mismatch
  - ConvNeXt-Tiny output stages don't match expected C3/C4/C5 naming
  - Memory usage exceeds RTX 3060 capacity

DEPENDS_ON: 1
