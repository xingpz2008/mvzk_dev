"""
================================================================================
MVZK Ground Truth Generator (Dual Mode)
================================================================================
Description:
  Loads the exact PyTorch model architecture and weights exported by the extractor.
  Supports BOTH standard torchvision models and custom user-defined scripts.
  Runs a standard plaintext forward pass on the provided input tensor (.npy).
  Exports the final output logits as a .bin file (FP32) for C++ ZK verification.

Usage Example (Mode 1: Vision Model):
  python ground_truth.py \
      --vision_model resnet18 \
      --weights_pth ../generated_model/resnet18_xxx/pytorch_weights.pth \
      --input_npy ../generated_model/resnet18_xxx/test_cases/case_1/input_0.npy \
      --output_dir ../generated_model/resnet18_xxx/test_cases/case_1

Usage Example (Mode 2: Custom Script):
  python ground_truth.py \
      --custom_script /path/to/my_net.py \
      --model_class SuperNet \
      --weights_pth ../generated_model/SuperNet_xxx/pytorch_weights.pth \
      --input_npy ../generated_model/SuperNet_xxx/test_cases/case_1/input_0.npy \
      --output_dir ../generated_model/SuperNet_xxx/test_cases/case_1
================================================================================
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import importlib.util
import sys
import ast

def is_script_safe_to_import(filepath: Path) -> tuple[bool, str]:
    """Static AST analysis to ensure no dangerous top-level execution exists."""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in script: {e}"

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Pass)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        elif isinstance(node, ast.If):
            try:
                if (isinstance(node.test, ast.Compare) and 
                    isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__' and
                    isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == '__main__'):
                    continue
            except Exception:
                pass
            return False, f"Unsafe top-level 'if' statement detected at line {node.lineno}."
        else:
            return False, f"Unsafe top-level execution detected ({type(node).__name__} at line {node.lineno})."

    return True, "Safe"

def generate_ground_truth(args):
    print("=================================================")
    print("      ZK-ML Ground Truth Generator (Oracle)      ")
    print("=================================================")

    # 1. Instantiate the model architecture (Dual Mode)
    model = None
    
    if args.vision_model:
        import torchvision.models as models
        print(f"[1] Instantiating torchvision model: {args.vision_model}...")
        model_class = getattr(models, args.vision_model)
        model = model_class(weights=None)
        
    elif args.custom_script and args.model_class:
        script_path = Path(args.custom_script).resolve()
        print(f"[1] Checking AST safety of custom script: {script_path.name}...")
        
        is_safe, msg = is_script_safe_to_import(script_path)
        if not is_safe:
            raise RuntimeError(f"[SECURITY ALERT] Import Rejected! Reason: {msg}")
            
        print("    -> AST safety check passed. Dynamically loading module...")
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            
        spec = importlib.util.spec_from_file_location("user_module", str(script_path))
        user_module = importlib.util.module_from_spec(spec)
        sys.modules["user_module"] = user_module
        spec.loader.exec_module(user_module)
        
        if sys.path[0] == script_dir:
            sys.path.pop(0)
            
        model_class = getattr(user_module, args.model_class)
        model = model_class()
    else:
        raise ValueError("[Error] Must specify either --vision_model OR (--custom_script AND --model_class)")

    model.eval()

    # =================================================================
    # [步骤调换] 先加载真实的测试输入数据！
    # =================================================================
    print(f"[2] Loading input tensor from: {args.input_npy}")
    input_np = np.load(args.input_npy)
    input_tensor = torch.from_numpy(input_np).float()
    print(f"    -> Input shape: {list(input_tensor.shape)}")

    # =================================================================
    # [核心补丁] 用真实的 input_tensor 触发网络结构的自适应重塑
    # =================================================================
    try:
        from extractor import auto_fold_adaptive_pool_and_linear
        auto_fold_adaptive_pool_and_linear(model, input_tensor)
    except ImportError:
        print("[WARNING] extractor.py not found, skipping adaptive pool folding.")

    # =================================================================
    # [加载权重] 现在的 model 已经完美变形，可以接住压缩后的权重了！
    # =================================================================
    print(f"[3] Loading precise extracted weights from: {args.weights_pth}")
    try:
        model.load_state_dict(torch.load(args.weights_pth, map_location='cpu'))
    except Exception as e:
        raise RuntimeError(f"Failed to load weights. Error: {e}")

    # 4. Execute the plaintext PyTorch forward pass
    print("[4] Running PyTorch Native Forward Pass...")
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    print(f"    -> Output shape: {list(output_tensor.shape)}")

    # 5. Export the expected logits (Standard Answer)
    expected_logits = output_tensor.detach().cpu().numpy().astype(np.float32)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bin_path = out_dir / "expected_logits.bin"
    
    expected_logits.tofile(str(out_bin_path))
    
    print("\n[SUCCESS] Ground Truth generated perfectly!")
    print(f"  - Target file saved to: {out_bin_path.resolve()}")
    
    # 动态打印所有 Batch 的 Argmax 预测结果（智能适配维度）
    if expected_logits.ndim >= 2:
        pred_classes = np.argmax(expected_logits, axis=1)
        for i, pred_class in enumerate(pred_classes):
            # 防止 Batch 太大刷屏，最多打印前 5 个
            if i >= 5:
                print(f"  - ... and {len(pred_classes) - 5} more items in the batch.")
                break
            print(f"  - [Oracle Prediction] Batch {i} -> Class Index: {pred_class}, Logit Value: {expected_logits[i, pred_class]:.4f}")
    else:
        # 兜底：如果输出是一维的
        pred_class = np.argmax(expected_logits)
        print(f"  - [Oracle Prediction] Class Index: {pred_class}, Logit Value: {expected_logits[pred_class]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ground Truth for ZK-ML Verification")
    
    # Mode 1: Built-in Vision Models
    parser.add_argument('--vision_model', type=str, default=None, help="Model architecture name (e.g., resnet18)")
    
    # Mode 2: Custom User Scripts
    parser.add_argument('--custom_script', type=str, default=None, help="Path to your python model script (e.g., my_net.py)")
    parser.add_argument('--model_class', type=str, default=None, help="Class name in your script (e.g., MyCustomNet)")
    
    # Required parameters
    parser.add_argument('--weights_pth', type=str, required=True, help="Path to the extracted pytorch_weights.pth")
    parser.add_argument('--input_npy', type=str, required=True, help="Path to the test case input_0.npy")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the expected_logits.bin")
    
    args = parser.parse_args()
    generate_ground_truth(args)