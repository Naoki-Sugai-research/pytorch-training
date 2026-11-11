import numpy as np
import torch

data = np.array([
[[85, 78], [67, 82], [92, 88], [75, 70], [60, 64]],
[[70, 68], [77, 72], [85, 90], [60, 65], [78, 76]],
[[80, 84], [88, 87], [66, 68], [72, 73], [64, 60]]
])

torch.tensor = torch.from_numpy(data)
print(torch.tensor)
torch.tensor = torch.permute(tensor, (2, 0, 1))
print(torch.tensor) 
torch.tensor = 