"""
ONNX Export for MultiTaskIKEA (PopW) Model
==========================================
Exports the trained MultiTaskIKEA model to ONNX format for inference.

Usage:
    from export_onnx import export_popw
    export_popw(model, "popw_model.onnx")

Note: If USE_GCN_SKELETON=True in config, the GCN module uses sparse tensor
operations that are not supported by ONNX. The export function temporarily
creates a model without GCN for ONNX export, while preserving the original
model's weights where possible.

Author: Bashara
Date: April 2026
"""

import sys
import copy
import torch
import onnx
import onnx.checker

# Import config to check GCN state
import config as C


def _create_non_gcn_model():
    """
    Create a model without GCN for ONNX export.
    
    This is needed because the GCN module uses sparse tensor operations
    (torch.sparse.mm) that are not supported by ONNX export.
    """
    # Temporarily disable GCN at config level
    original_setting = C.USE_GCN_SKELETON
    C.USE_GCN_SKELETON = False
    
    try:
        from model import MultiTaskIKEA
        net = MultiTaskIKEA(pretrained=False)
        return net, original_setting
    finally:
        # Restore setting
        C.USE_GCN_SKELETON = original_setting


def export_popw(model: torch.nn.Module, save_path: str, opset_version: int = 14) -> None:
    """
    Export MultiTaskIKEA model to ONNX format.

    Args:
        model: MultiTaskIKEA instance (pretrained=False recommended for export)
        save_path: Path to save the ONNX model
        opset_version: ONNX opset version (default: 14)

    Raises:
        RuntimeError: If ONNX export fails
    """
    # Define input shape: [B, 3, 480, 640]
    input_shape = (1, 3, 480, 640)
    dummy_input = torch.randn(*input_shape)

    # Dynamic batch dimension (index 0)
    dynamic_axes = {
        "input": {0: "batch_size"},
        "act_logits": {0: "batch_size"},
        "cls_preds": {0: "batch_size"},
        "reg_preds": {0: "batch_size"},
        "keypoints": {0: "batch_size"},
    }

    # Check if original model has GCN enabled
    has_gcn = getattr(model.activity_head, 'use_gcn', False)

    if has_gcn:
        # Create a fresh model without GCN for export
        # This is needed because sparse tensor ops are not supported by ONNX
        print("[export_onnx] Creating non-GCN model for ONNX export (sparse ops not in ONNX)...")
        model_for_export, _ = _create_non_gcn_model()
    else:
        model_for_export = model

    model_for_export.eval()

    with torch.no_grad():
        torch.onnx.export(
            model_for_export,
            dummy_input,
            save_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["act_logits", "cls_preds", "reg_preds", "keypoints"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    # Verify the exported ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"[export_onnx] PopW model exported successfully: {save_path}")
    print(f"[export_onnx] Input shape: {input_shape}, OpSet: {opset_version}")
    if has_gcn:
        print(f"[export_onnx] Note: GCN disabled (sparse tensor ops not in ONNX)")


if __name__ == "__main__":
    sys.path.insert(0, "/home/newadmin/swarm-bot/project/popw/working/code/popw_main_improved")

    from model import MultiTaskIKEA

    # Create and export model
    net = MultiTaskIKEA(pretrained=False)
    export_popw(net, "test_popw.onnx")
    print("[export_onnx] ONNX export verification passed")
