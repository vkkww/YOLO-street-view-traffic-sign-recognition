import torch
import torchvision
from torchvision import ops

print("TorchVision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Testing NMS on GPU...")
    # 注意这里加了 .float() 来确保 boxes 是浮点类型
    boxes = torch.tensor([[0, 0, 100, 100], [10, 10, 110, 110]], device="cuda").float()
    scores = torch.tensor([0.9, 0.8], device="cuda")

    try:
        keep = ops.nms(boxes, scores, iou_threshold=0.5)
        print("Kept indices:", keep)
    except Exception as e:
        print("Error during NMS:", str(e))
else:
    print("CUDA is not available, skipping GPU test.")
