import torch
from networks.dan import DAN, CenterLoss, PartitionLoss
import sys

def verify():
    print("Verifying DAN implementation...")
    
    # 1. Instantiate Model
    try:
        model = DAN(num_class=7, num_head=4, pretrained=False)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    # 2. Dummy Input
    x = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {x.shape}")

    # 3. Forward Pass
    try:
        out, feat, heads = model(x)
        print("Forward pass successful.")
        print(f"Output shape (logits): {out.shape}")
        print(f"Features shape: {feat.shape}")
        print(f"Heads shape: {heads.shape}")
        
        assert out.shape == (2, 7), f"Expected (2, 7), got {out.shape}"
        assert feat.shape == (2, 512), f"Expected (2, 512), got {feat.shape}"
        assert heads.shape == (2, 4, 512), f"Expected (2, 4, 512), got {heads.shape}"
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return

    # 4. Loss Calculation
    try:
        targets = torch.tensor([0, 1])
        
        criterion_cls = torch.nn.CrossEntropyLoss()
        criterion_center = CenterLoss(num_classes=7, feat_dim=512, use_gpu=False)
        criterion_pt = PartitionLoss()
        
        loss_cls = criterion_cls(out, targets)
        loss_center = criterion_center(feat, targets)
        loss_pt = criterion_pt(heads)
        
        total_loss = loss_cls + 0.1 * loss_pt + 0.003 * loss_center
        print(f"Total loss: {total_loss.item()}")
        
    except Exception as e:
        print(f"Loss calculation failed: {e}")
        return

    # 5. Backward Pass
    try:
        total_loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return

    print("\nVerification Passed!")

if __name__ == "__main__":
    verify()
