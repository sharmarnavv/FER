import torch
import torch.nn as nn
from networks.dan import DAN
import torchvision.models as models
import time
import numpy as np

def test_backbone(backbone_name):
    print(f"\n--- Testing backbone: {backbone_name} ---")
    try:
        model = DAN(num_class=7, num_head=4, pretrained=False, backbone=backbone_name)
        print(f"Model loaded. Backbone type: {type(model.backbone)}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Create dummy input
        batch_size = 16
        x = torch.randn(batch_size, 3, 224, 224)
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # Timing
        start_time = time.time()
        iterations = 50
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        fps = batch_size / avg_time
        
        print(f"Forward pass successful.")
        print(f"Average batch time: {avg_time*1000:.2f}ms")
        print(f"Throughput: {fps:.2f} images/sec")
        
        return {
            "backbone": backbone_name,
            "params": total_params,
            "latency_ms": avg_time * 1000,
            "throughput_fps": fps
        }
    except Exception as e:
        print(f"Failed to load/run model: {e}")
        return None

if __name__ == "__main__":
    results = []
    
    # Test ResNet18
    res = test_backbone('resnet18')
    if res: results.append(res)
    
    # Test ConvNeXt Tiny
    res = test_backbone('convnext_tiny')
    if res: results.append(res)
    
    print("\n\n=== Comparison Results ===")
    print(f"{'Backbone':<20} | {'Params':<15} | {'Latency (ms)':<15} | {'FPS':<15}")
    print("-" * 75)
    for r in results:
        print(f"{r['backbone']:<20} | {r['params']:<15,} | {r['latency_ms']:<15.2f} | {r['throughput_fps']:<15.2f}")
