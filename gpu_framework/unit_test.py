from utils import select_argmax
import torch

torch.manual_seed(0)
interval_right = torch.sigmoid(torch.randn(6, 3))
interval_left = interval_right - torch.sigmoid(torch.randn(6, 3)) * 0.1
interval_left[interval_left < 0] = 0
if torch.cuda.is_available():
    interval_right = interval_right.cuda()
    interval_left = interval_left.cuda()

print(f"interval_right:\n{interval_right}\ninterval_left:\n{interval_left}")
index_mask = select_argmax(interval_left, interval_right)
print(f"index_mask:\n{index_mask}")