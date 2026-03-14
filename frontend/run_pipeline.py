"""
================================================================================
ZK-ML End-to-End Automation Pipeline
================================================================================
Usage:
  python run_pipeline.py --vision_model resnet18 --height 32 --width 32
  python run_pipeline.py --custom_script my_net.py --model_class SuperNet
================================================================================
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# --- 终端颜色配置 ---
C_CYAN = "\033[1;36m"
C_GREEN = "\033[1;32m"
C_YELLOW = "\033[1;33m"
C_RED = "\033[1;31m"
C_RESET = "\033[0m"

def run_cmd(cmd_list, step_name):
    print(f"\n{C_CYAN}================================================={C_RESET}")
    print(f"{C_CYAN}[STEP] {step_name}{C_RESET}")
    print(f"{C_CYAN}================================================={C_RESET}")
    cmd_str = " ".join(cmd_list)
    print(f"{C_YELLOW}> {cmd_str}{C_RESET}\n")
    
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"\n{C_RED}[FATAL] Pipeline aborted at {step_name}!{C_RESET}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZK-ML Automated Pipeline")
    parser.add_argument('--vision_model', type=str, default=None)
    parser.add_argument('--custom_script', type=str, default=None)
    parser.add_argument('--model_class', type=str, default=None)
    
    # 只需要在这里输入一次尺寸，全宇宙都会知道！
    parser.add_argument('--batch', type=str, default="1")
    parser.add_argument('--channels', type=str, default="3")
    parser.add_argument('--height', type=str, default="32")
    parser.add_argument('--width', type=str, default="32")
    
    args = parser.parse_args()

    # 1. 提取器 (Extractor)
    extractor_cmd = [sys.executable, "extractor.py", 
                     "--batch", args.batch, "--channels", args.channels, 
                     "--height", args.height, "--width", args.width]
    
    if args.vision_model:
        extractor_cmd.extend(["--vision_model", args.vision_model])
        model_name = args.vision_model
    elif args.custom_script and args.model_class:
        extractor_cmd.extend(["--custom_script", args.custom_script, "--model_class", args.model_class])
        model_name = args.model_class
    else:
        print(f"{C_RED}Please specify --vision_model OR (--custom_script and --model_class){C_RESET}")
        sys.exit(1)
        
    run_cmd(extractor_cmd, "Model Extraction & Graph Generation")

    # 自动获取刚刚生成的带有时间戳的文件夹
    base_dir = Path(__file__).resolve().parent.parent / "generated_model"
    # 按照修改时间排序，拿到最新生成的那个文件夹
    out_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and model_name in d.name], key=os.path.getmtime)
    if not out_dirs:
        print(f"{C_RED}[FATAL] Cannot find generated model directory!{C_RESET}")
        sys.exit(1)
        
    target_dir = str(out_dirs[-1])
    test_case_dir = str(out_dirs[-1] / "test_cases" / "case_1")
    
    # 2. 输入数据生成 (Input Processor) - 不用传尺寸了，它会自动读 metadata.json！
    input_cmd = [sys.executable, "input_processor.py", "--target_dir", test_case_dir]
    run_cmd(input_cmd, "Dummy Data Generation (Auto-SSOT)")

    # 3. 标准答案生成 (Ground Truth)
    gt_cmd = [sys.executable, "ground_truth.py", 
              "--weights_pth", str(Path(target_dir) / "pytorch_weights.pth"),
              "--input_npy", str(Path(test_case_dir) / "input_0.npy"),
              "--output_dir", test_case_dir]
              
    if args.vision_model:
        gt_cmd.extend(["--vision_model", args.vision_model])
    else:
        gt_cmd.extend(["--custom_script", args.custom_script, "--model_class", args.model_class])
        
    run_cmd(gt_cmd, "Oracle Ground Truth Calculation")

    # 4. ZK C++ 电路编译 (Compiler)
    compiler_cmd = [sys.executable, "compiler.py", 
                    "--json", str(Path(target_dir) / "graph.json"),
                    "--model_name", model_name]
    run_cmd(compiler_cmd, "AOT ZK-Circuit Compilation")

    print(f"\n{C_GREEN}================================================={C_RESET}")
    print(f"{C_GREEN} PIPELINE FINISHED SUCCESSFULLY {C_RESET}")
    print(f"{C_GREEN}================================================={C_RESET}")
    
    # [体验优化] 提取相对路径，假设用户接下来会去 build 目录执行
    folder_name = Path(target_dir).name
    rel_model_dir = f"../generated_model/{folder_name}"
    
    print(f"Generated C++ files are located at: {C_YELLOW}{rel_model_dir}{C_RESET}")
    print(f"\nNext Steps:")
    print(f"1. cd ../build")
    print(f"2. cmake .. -DTEST_SELECT=GEN && make -j")
    print(f"3. cd ../generated_model/{folder_name}")
    print(f"4. ./{model_name}_test 1 12345 .")
    print(f"   ./{model_name}_test 2 12345 .\n")