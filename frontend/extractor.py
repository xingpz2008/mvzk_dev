"""
================================================================================
ZK-ML PyTorch Model Exporter (AST-Secured)
================================================================================

This is a customized PyTorch model graph extraction and weight export tool 
designed for Zero-Knowledge Machine Learning (ZK-ML) engines. Its primary 
function is to convert PyTorch models into a friendly Intermediate Representation 
(IR) for the ZK-C++ backend.

Key Features:
1. Extracts the static computational graph topology and saves it as `graph.json`.
2. Flattens tensor weights and biases, saves them as `.bin` binary files, and 
   records their shapes in the JSON.
3. Offline graph optimization: Automatically folds BatchNorm layers 
   (Fuses Conv2d + BatchNorm2d).
4. ZK-specific optimization: Automatically identifies and fuses ReLU + MaxPool2d 
   into an `integrated_nl` operator.
5. Industrial-grade security: Utilizes static AST analysis to intercept dangerous 
   user script injections.

Dependencies:
    pip install torch torchvision numpy

Arguments:
    --out_dir       : (Optional) Destination directory path for the exported model.
                      If not provided, it defaults to:
                      ../generated_model/<model_name>_<timestamp>/
    --vision_model  : (Mode 1) Name of a standard built-in torchvision model 
                      (e.g., vgg16_bn, resnet18).
    --custom_script : (Mode 2) Path to the user-defined .py file containing 
                      the model architecture.
    --model_class   : (Mode 2) Name of the core model class inside the 
                      user-defined .py file.
    --weights       : (Optional) Path to the user's pre-trained .pth weights 
                      file. If not provided, random initialized weights are used.

Usage Examples:

    Scenario 1: Export a standard torchvision model (Output to default timestamped dir)
        python extract.py --vision_model vgg16_bn

    Scenario 2: Export a custom model (Output to default timestamped dir)
        python extract.py \
            --custom_script my_awesome_net.py \
            --model_class SuperNet

    Scenario 3: Export a custom model with weights to a specific custom directory
        python extract.py \
            --custom_script /path/to/my_awesome_net.py \
            --model_class SuperNet \
            --weights /path/to/epoch_100.pth \
            --out_dir /absolute/path/to/custom_dir

Security & Formatting Notice:
    When importing custom files using `--custom_script`, strict AST (Abstract 
    Syntax Tree) checking is built-in to prevent accidental code execution.
    Please ensure that your custom `.py` file does NOT contain any execution 
    logic (such as training loops, data loading, etc.) directly exposed at the 
    top level. All testing or execution code must be wrapped within the standard 
    entry block:
    
        if __name__ == "__main__":
            # Execution logic goes here

================================================================================
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace, Node, GraphModule
import json
import numpy as np
import argparse
import importlib.util
import sys
import ast
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. AST Security Checker
# ==========================================
def is_script_safe_to_import(filepath: Path) -> tuple[bool, str]:
    """
    Static AST analysis to ensure no dangerous top-level execution exists.
    """
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
            return False, f"Unsafe top-level 'if' statement detected at line {node.lineno}. Only 'if __name__ == \"__main__\":' is allowed."
        else:
            return False, f"Unsafe top-level execution detected ({type(node).__name__} at line {node.lineno}). Only definitions and assignments are allowed at the module level."

    return True, "Safe"


# ==========================================
# 2. BatchNorm Folding Logic & TRACING INTERCEPTOR
# ==========================================
def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    assert not (conv.training or bn.training), "Fusion only works in eval mode."
    gamma = bn.weight
    beta = bn.bias
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.groups, bias=True
    )
    w_conv = conv.weight.clone()
    scale_reshaped = scale.view(-1, 1, 1, 1)
    fused_conv.weight.data = w_conv * scale_reshaped
    b_conv = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels, device=w_conv.device)
    fused_conv.bias.data = (b_conv - mu) * scale + beta
    return fused_conv

def fx_fold_batchnorm(model: nn.Module) -> GraphModule:
    model.eval()
    
    try:
        traced_model = symbolic_trace(model)
    except Exception as e:
        print("\n" + "="*60)
        print("[CRITICAL ERROR] Model Tracing Failed!")
        print("="*60)
        print("We detected dynamic control flow inside your model's forward() function.")
        print("This is usually caused by data-dependent 'if' statements or 'for' loops.")
        print("\nOur ZK Exporter strictly requires a STATIC computational graph.")
        print("Please refactor your forward() pass or ensure the model architecture is static.")
        print(f"\n[Detailed PyTorch Trace Error]:\n{e}")
        print("="*60)
        sys.exit(1)
        
    modules = dict(traced_model.named_modules())
    
    for node in traced_model.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], nn.BatchNorm2d):
            prev_node = node.args[0]
            if prev_node.op == 'call_module' and isinstance(modules[prev_node.target], nn.Conv2d):
                conv_name = prev_node.target
                bn_name = node.target
                print(f"[Optimizer] Fusing BN: {conv_name} -> {bn_name}")
                
                fused_conv = fuse_conv_bn_eval(modules[conv_name], modules[bn_name])
                setattr(traced_model, conv_name, fused_conv)
                modules[conv_name] = fused_conv
                
                node.replace_all_uses_with(prev_node)
                traced_model.graph.erase_node(node)

    traced_model.recompile()
    return traced_model


# ==========================================
# 3. Graph Topology & Weights Exporter
# ==========================================
def export_zk_model(model: GraphModule, export_dir_str: str):
    export_dir = Path(export_dir_str)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dir = export_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    modules = dict(model.named_modules())
    zk_graph_ir = []

    for node in model.graph.nodes:
        if node.op == 'placeholder':
            zk_graph_ir.append({"name": node.name, "type": "input", "inputs": []})

        elif node.op == 'call_module':
            target_mod = modules[node.target]

            if isinstance(target_mod, nn.ReLU):
                users = list(node.users.keys())
                if len(users) == 1 and users[0].op == 'call_module' and isinstance(modules[users[0].target], nn.MaxPool2d):
                    print(f"[Optimizer] Skipping ReLU '{node.name}', merging with MaxPool2d.")
                    continue  
                else:
                    zk_graph_ir.append({"name": node.name, "type": "relu", "inputs": [str(node.args[0])]})

            elif isinstance(target_mod, nn.MaxPool2d):
                prev_node = node.args[0]
                is_fused = False
                
                # [BUG FIX] strict fusion check: ReLU must have EXACTLY 1 user (this MaxPool2d)
                if prev_node.op == 'call_module' and isinstance(modules[prev_node.target], nn.ReLU) and len(prev_node.users) == 1:
                    is_fused = True
                    prev_prev_node = prev_node.args[0] 
                    print(f"[Optimizer] Exporting fused '{node.name}' as integrated_nl.")
                    zk_graph_ir.append({
                        "name": node.name, "type": "integrated_nl",
                        "inputs": [str(prev_prev_node)], 
                        "kernel_size": target_mod.kernel_size, 
                        "stride": target_mod.stride,
                        "padding": target_mod.padding
                    })
                
                if not is_fused: 
                    zk_graph_ir.append({
                        "name": node.name, "type": "max_pool2d", "inputs": [str(node.args[0])],
                        "kernel_size": target_mod.kernel_size, "stride": target_mod.stride
                    })

            elif isinstance(target_mod, (nn.Conv2d, nn.Linear)):
                # 1. 处理带有权重和偏置的参数化算子 (Conv2d, Linear)
                node_type = type(target_mod).__name__.lower()
                node_info = {"name": node.name, "type": node_type, "inputs": [str(node.args[0])]}
                
                if isinstance(target_mod, nn.Conv2d):
                    node_info["stride"] = target_mod.stride
                    node_info["padding"] = target_mod.padding
                
                # 导出权重
                if hasattr(target_mod, 'weight') and target_mod.weight is not None:
                    node_info["weight_shape"] = list(target_mod.weight.shape)
                    w_path = weights_dir / f"{node.name}_weight.bin"
                    target_mod.weight.detach().cpu().numpy().astype(np.float32).tofile(str(w_path))
                
                # 导出偏置
                if hasattr(target_mod, 'bias') and target_mod.bias is not None:
                    node_info["bias_shape"] = list(target_mod.bias.shape)
                    b_path = weights_dir / f"{node.name}_bias.bin"
                    target_mod.bias.detach().cpu().numpy().astype(np.float32).tofile(str(b_path))
                    
                zk_graph_ir.append(node_info)

            elif isinstance(target_mod, nn.AdaptiveAvgPool2d):
                # 2. 专项处理自适应池化层
                out_size = target_mod.output_size
                
                # 判断是否为严格的全局平均池化 (1x1)
                is_gap = (out_size == (1, 1) or out_size == 1 or out_size == [1, 1])
                
                if is_gap:
                    # 将算子类型显式重命名为 global_avg_pool2d，方便 C++ 模板识别
                    node_info = {"name": node.name, "type": "global_avg_pool2d", "inputs": [str(node.args[0])]}
                    zk_graph_ir.append(node_info)
                else:
                    # 防御性拦截：不支持非 1x1 的 AdaptiveAvgPool
                    print(f"\n[CRITICAL ERROR] AdaptiveAvgPool2d with output_size={out_size} is detected!")
                    print("The ZK C++ backend currently ONLY supports AdaptiveAvgPool2d when used as a Global Average Pooling (output_size=1).")
                    print("Please modify your PyTorch model architecture.")
                    sys.exit(1)

            else:
                module_type = type(target_mod).__name__.lower()
                print(f"[Warning] Exporting generic/unsupported module: '{node.name}' (Type: {module_type})")
                
                node_info = {
                    "name": node.name,
                    "type": f"module_{module_type}", 
                    "inputs": [str(arg) for arg in node.args if isinstance(arg, Node)],
                    "status": "UNCHECKED",
                    "param_shapes": {} # [FEATURE] Track shapes for unknown modules
                }
                
                for param_name, param_tensor in target_mod.named_parameters():
                    if param_tensor is not None:
                        node_info["param_shapes"][param_name] = list(param_tensor.shape)
                        bin_filename = f"{node.name}_{param_name.replace('.', '_')}.bin"
                        bin_path = weights_dir / bin_filename
                        param_tensor.detach().cpu().numpy().astype(np.float32).tofile(str(bin_path))
                
                zk_graph_ir.append(node_info)

        elif node.op == 'call_function':
            func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            
            if func_name == "flatten":
                zk_graph_ir.append({"name": node.name, "type": "flatten", "inputs": [str(node.args[0])]})
            else:
                zk_graph_ir.append({
                    "name": node.name,
                    "type": f"func_{func_name}", 
                    "inputs": [str(arg) for arg in node.args if isinstance(arg, Node)],
                    "args_const": [str(arg) for arg in node.args if not isinstance(arg, Node)],
                    "kwargs": {k: str(v) for k, v in node.kwargs.items()} 
                })

        elif node.op == 'call_method':
            method_name = str(node.target)
            tensor_input = node.args[0]
            other_args = [str(arg) for arg in node.args[1:] if not isinstance(arg, Node)]
            
            zk_graph_ir.append({
                "name": node.name,
                "type": f"method_{method_name}", 
                "inputs": [str(tensor_input)] if isinstance(tensor_input, Node) else [],
                "method_args": other_args 
            })

        elif node.op == 'output':
            out_input = node.args[0][0] if isinstance(node.args[0], tuple) else node.args[0]
            zk_graph_ir.append({"name": node.name, "type": "output", "inputs": [str(out_input)]})

    json_path = export_dir / "graph.json"
    with open(json_path, "w") as f:
        json.dump(zk_graph_ir, f, indent=4)
        
    print(f"\n[Success] Model exported successfully to directory: {export_dir.resolve()}")
    print(f"- Topology saved at: {json_path.resolve()}")
    print(f"- Binary weights saved at: {weights_dir.resolve()}")


# ==========================================
# 4. Main CLI Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZK-ML Model Exporter (AST-Secured)")
    parser.add_argument('--vision_model', type=str, help="Name of torchvision model (e.g., vgg16_bn)")
    parser.add_argument('--custom_script', type=str, help="Path to your python model script (e.g., my_net.py)")
    parser.add_argument('--model_class', type=str, help="Class name in your script (e.g., MyCustomNet)")
    parser.add_argument('--weights', type=str, help="Path to the .pth state_dict file", default=None)
    parser.add_argument('--out_dir', type=str, default=None, help="Optional specific output directory")
    
    args = parser.parse_args()
    model = None
    model_name = "unknown_model"

    if args.vision_model:
        import torchvision.models as models
        print(f"Loading standard torchvision model: {args.vision_model}")
        model_fn = getattr(models, args.vision_model)
        model = model_fn(weights=None)
        model_name = args.vision_model
        
    elif args.custom_script and args.model_class:
        script_path = Path(args.custom_script).resolve()
        
        print(f"Checking AST safety of {script_path.name}...")
        is_safe, msg = is_script_safe_to_import(script_path)
        if not is_safe:
            print(f"\n[SECURITY ALERT] Import Rejected!")
            print(f"Reason: {msg}")
            print("Please wrap any execution/training code inside 'if __name__ == \"__main__\":'")
            sys.exit(1)
            
        print("AST safety check passed. Dynamically loading module...")
        
        # [FEATURE] Dynamic sys.path injection to resolve local dependencies
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            
        spec = importlib.util.spec_from_file_location("user_module", str(script_path))
        user_module = importlib.util.module_from_spec(spec)
        sys.modules["user_module"] = user_module
        spec.loader.exec_module(user_module)
        
        # Cleanup sys.path (optional but good practice)
        if sys.path[0] == script_dir:
            sys.path.pop(0)
            
        model_class = getattr(user_module, args.model_class)
        model = model_class() 
        model_name = args.model_class
    else:
        print("[Error] Invalid arguments. Specify either --vision_model OR (--custom_script AND --model_class)")
        sys.exit(1)

    if args.weights:
        print(f"Loading weights from {args.weights}...")
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    print("\n[Step 1] Initializing Graph Optimization (BatchNorm Folding)...")
    fused_net = fx_fold_batchnorm(model)
    
# [FEATURE] Dynamic timestamped output directory generation
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 直接定位到 emp-mvzk/generated_model
        base_dir = Path(__file__).resolve().parent.parent / "generated_model"
        final_out_dir = str(base_dir / f"{model_name}_{timestamp}")
    else:
        final_out_dir = args.out_dir

    print(f"\n[Step 2] Exporting JSON graph and .bin weights to {final_out_dir}...")
    export_zk_model(fused_net, export_dir_str=final_out_dir)