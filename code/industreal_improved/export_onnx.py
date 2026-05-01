"""
ONNX Export for MultiTaskIndustReal Model
==========================================
Exports the trained MultiTaskIndustReal model to ONNX format for inference.

Usage:
    from export_onnx import export_industreal
    export_industreal(model, "industreal_model.onnx")

Author: Bashara
Date: April 2026
"""

import sys
import torch
import onnx
import onnx.checker


def export_industreal(model: torch.nn.Module, save_path: str, opset_version: int = 14) -> None:
    """
    Export MultiTaskIndustReal model to ONNX format.

    Args:
        model: MultiTaskIndustReal instance (pretrained=False recommended for export)
        save_path: Path to save the ONNX model
        opset_version: ONNX opset version (default: 14)

    Raises:
        RuntimeError: If ONNX export fails
    """
    # Set model to eval mode (disables dropout, uses running stats for BN)
    model.eval()

    # Define input shape: [B, 3, 720, 1280]
    # Batch dimension (index 0) is set as dynamic for flexibility
    input_shape = (1, 3, 720, 1280)
    dummy_input = torch.randn(*input_shape)

    # Dynamic batch dimension (index 0)
    dynamic_axes = {
        "input": {0: "batch_size"},
        "act_logits": {0: "batch_size"},
        "cls_preds": {0: "batch_size"},
        "reg_preds": {0: "batch_size"},
        "head_pose": {0: "batch_size"},
        "psr_logits": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["act_logits", "cls_preds", "reg_preds", "head_pose", "psr_logits"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    # Verify the exported ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"[export_onnx] IndustReal model exported successfully: {save_path}")
    print(f"[export_onnx] Input shape: {input_shape}, OpSet: {opset_version}")


if __name__ == "__main__":
    sys.path.insert(0, "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved")

    from model import POPWMultiTaskModel

    net = POPWMultiTaskModel(
        pretrained=False,
        backbone_type='convnext_tiny',
        use_headpose_film=True,
        use_videomae=False,
    )
    export_industreal(net, "test_industreal.onnx")
    print("[export_onnx] ONNX export verification passed")
