## Execution Order
Serial (must run in sequence):
  1. Contract #1: Create popw_main_improved folder structure
  2. Contract #2: Implement ConvNeXt-Tiny backbone for PopW (depends on #1)
  3. Contract #3: Implement OKS Loss for PopW (depends on #1)
  4. Contract #4: Implement GCN skeleton for PopW (depends on #1)
  5. Contract #5: Create industreal_improved folder structure
  6. Contract #6: Implement ConvNeXt-Tiny backbone for IndustReal (depends on #5)
  7. Contract #7: Implement TMA Cell for IndustReal (depends on #5)
  8. Contract #8: Implement Temporal Bank for IndustReal (depends on #5)

Parallel (can run simultaneously):
  - Contracts #2, #3, #4 can run in parallel after #1 completes
  - Contracts #6, #7, #8 can run in parallel after #5 completes

Final gate (must run last):
  - Integration test: full forward pass with all improvements enabled
  - Parameter count verification: < 49M total
  - FPS benchmark: > 291 FPS on RTX 3060

## Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ConvNeXt-Tiny FPN channel mismatch | H | H | Verify channel counts before FPN integration |
| timm not installed or ConvNeXt unavailable | M | H | Pre-check `pip install timm` |
| OKS Loss gradient instability | M | M | Add gradient clipping, use smooth L1 |
| GCN adjacency matrix causes gradient explosion | M | M | Laplacian normalization, gradient clipping |
| TMA Cell memory usage with T=32 | H | H | Use frame subsampling, gradient checkpointing |
| Temporal Bank breaks dataloader compatibility | M | H | Ensure dataset returns List[Tensor], not 5D tensor |
| RTX 3060 OOM with all improvements | H | H | Reduce batch size, use gradient accumulation |
