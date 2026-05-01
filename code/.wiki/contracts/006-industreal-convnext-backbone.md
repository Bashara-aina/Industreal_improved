### CONTRACT #6: Implement ConvNeXt-Tiny backbone for IndustReal

WHAT:
  Replace ResNet-50 backbone with ConvNeXt-Tiny in industreal_improved/model.py, maintaining FPN compatibility with channel outputs C3(192ch), C4(384ch), C5(768ch).

FILES:
  READ:  /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
  WRITE: /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/model.py
  RUN:   python -c "import timm; m = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=False); print('ConvNeXt-Tiny available')"

DONE_WHEN:
  - ConvNeXtBackbone class replaces ResNet50 in model.py
  - FPN initialized with in_channels_list=[192, 384, 768] for ConvNeXt-Tiny stages 2/3/4
  - Backbone uses `fb_in22k_ft_in1k` pretrained
  - Frozen BN strategy maintained (eval mode + requires_grad=False)
  - Total model params < 49M
  - ActivityHead, HeadPoseHead, PSRHead, DetectionHead unchanged interfaces

PROOF_FORMAT:
  CODE: `cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved && python -c "
import torch, model
net = model.MultiTaskIndustReal(pretrained=False)
params = sum(p.numel() for p in net.parameters() if p.requires_grad)
total = sum(p.numel() for p in net.parameters())
print(f'Trainable: {params:,}, Total: {total:,}')
x = torch.randn(1, 3, 720, 1280)
out = net(x)
print(f'Output keys: {list(out.keys())}')
"` → should print params and output keys

BLOCKER_IF:
  - FPN channel mismatch with ConvNeXt-Tiny outputs
  - Memory exceeds RTX 3060 capacity during forward pass

DEPENDS_ON: 5
