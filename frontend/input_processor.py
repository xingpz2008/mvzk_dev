"""
================================================================================
MVZK Input Data Processor (with Quantization Support)
================================================================================
Description:
  Standalone data processing module for the MVZK pipeline.
  Supports fetching standard datasets, real images, or generating dummy data.
  
  [NEW] Quantization Support:
  Can optionally quantize the FP32 input tensor to INT8 before exporting, 
  which is crucial for testing quantized ZK models.

Usage Examples:
  1. Standard FP32 Dummy Data:
     python input_processor.py --target_dir ./out
     
  2. FP32 Real Image (Batch=4):
     python input_processor.py --target_dir ./out --image_path ./cat.jpg --batch 4
     
  3. [NEW] INT8 Quantized Input from CIFAR-10 (Requires scale and zero_point):
     python input_processor.py --target_dir ./out --dataset cifar10 --batch 128 \
         --quantize --scale 0.015 --zero_point 0
================================================================================
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import json

try:
    from PIL import Image
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

def process_and_save_input(args):
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # [状态管理] 智能向上寻找 metadata.json（兼容流水线和手动执行）
    meta_path = None
    current_search_dir = target_dir.resolve()
    
    # 最多向上找 3 级目录
    for _ in range(3):
        candidate = current_search_dir / "metadata.json"
        if candidate.exists():
            meta_path = candidate
            break
        current_search_dir = current_search_dir.parent

    if meta_path:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            args.batch = meta.get("N", args.batch)
            args.channels = meta.get("C", args.channels)
            args.height = meta.get("H", args.height)
            args.width = meta.get("W", args.width)
        print(f"\033[1;36m[Input Processor]\033[0m Auto-loaded dimensions from metadata: {args.batch}x{args.channels}x{args.height}x{args.width}")
    else:
        print(f"\033[33m[WARNING] metadata.json not found in parents of {target_dir}. Using default arguments.\033[0m")
    
    tensor_shape = (args.batch, args.channels, args.height, args.width)
    labels_tensor = None 
    
    # ==========================================
    # Mode C: Load from standard torchvision datasets
    # ==========================================
    if args.dataset:
        if not HAS_VISION:
            raise ImportError("[Error] torchvision is required to load standard datasets.")
            
        print(f"[Input Processor] Fetching 1 batch from standard dataset: {args.dataset.upper()}")
        
        transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset_name = args.dataset.lower()
        root_dir = './frontent_dataset' 
        
        if dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
        elif dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)
        else:
            raise ValueError(f"[Error] Unsupported dataset: {dataset_name}.")
            
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
        data_iter = iter(dataloader)
        
        input_tensor, labels_tensor = next(data_iter)
        print(f"  -> Fetched {args.batch} samples. Input shape: {list(input_tensor.shape)}")

    # ==========================================
    # Mode A: Process a real local image
    # ==========================================
    elif args.image_path:
        if not HAS_VISION:
            raise ImportError("[Error] Pillow and torchvision are required to process real images.")
        
        print(f"[Input Processor] Processing local image: {args.image_path}")
        img = Image.open(args.image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(img).unsqueeze(0)
        
        if args.batch > 1:
            input_tensor = input_tensor.repeat(args.batch, 1, 1, 1)
            
        print(f"  -> Applied vision transforms. Final shape: {list(input_tensor.shape)}")

    # ==========================================
    # Mode B: Generate dummy test data
    # ==========================================
    else:
        print(f"[Input Processor] Generating dummy input tensor with shape {tensor_shape}...")
        torch.manual_seed(42) 
        input_tensor = torch.rand(*tensor_shape) - 0.5
        print("  -> Dummy data generated.")

    # ==========================================
    # [FEATURE] Quantization Processing
    # ==========================================
    if args.quantize:
        print(f"\n[Input Processor] Applying INT8 Quantization...")
        print(f"  -> Scale: {args.scale}, Zero Point: {args.zero_point}")
        
        # Apply the standard quantization formula: X_int = round(X_float / scale) + zero_point
        quantized_tensor = torch.round(input_tensor / args.scale) + args.zero_point
        
        # Clamp to 8-bit signed integer range [-128, 127] and cast
        quantized_tensor = torch.clamp(quantized_tensor, -128, 127).to(torch.int8)
        
        # Extract as contiguous int8 array
        np_array = quantized_tensor.detach().cpu().numpy().astype(np.int8)
        dtype_str = "INT8"
    else:
        # Default: Keep as FP32
        np_array = input_tensor.detach().cpu().numpy().astype(np.float32)
        dtype_str = "FP32"

    # ==========================================
    # Export Data
    # ==========================================
    bin_path = target_dir / "input_0.bin"
    npy_path = target_dir / "input_0.npy"
    
    np_array.tofile(str(bin_path))
    np.save(str(npy_path), np_array)
    
    print(f"\n[SUCCESS] Input processing complete!")
    print(f"  - Input C++ Binary ({dtype_str}) saved to : {bin_path.resolve()}")
    print(f"  - Input Python Numpy ({dtype_str}) saved to: {npy_path.resolve()}")
    
    # Export labels if dataset was used
    if labels_tensor is not None:
        labels_npy_path = target_dir / "labels_0.npy"
        np.save(str(labels_npy_path), labels_tensor.detach().cpu().numpy())
        print(f"  - Labels Python Numpy saved to: {labels_npy_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or process input data for ZK-ML testing.")
    
    parser.add_argument('--target_dir', type=str, required=True, help="Directory to save the processed input files")
    
    # Target dimension settings
    parser.add_argument('--batch', type=int, default=1, help="Batch size (can be > 1)")
    parser.add_argument('--channels', type=int, default=3, help="Number of channels")
    parser.add_argument('--height', type=int, default=224, help="Image height")
    parser.add_argument('--width', type=int, default=224, help="Image width")
    
    # Input source options
    parser.add_argument('--image_path', type=str, default=None, help="Path to a single real image file")
    parser.add_argument('--dataset', type=str, default=None, help="Name of torchvision dataset (e.g., cifar10, mnist)")
    
    # Quantization options
    parser.add_argument('--quantize', action='store_true', help="Enable INT8 quantization for the output tensor")
    parser.add_argument('--scale', type=float, default=1.0, help="Quantization scale factor")
    parser.add_argument('--zero_point', type=int, default=0, help="Quantization zero point")
    
    args = parser.parse_args()
    
    process_and_save_input(args)