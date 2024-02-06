import torch

# 检查 GPU 是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')  # 指定 GPU 设备
    print('GPU is available')
else:
    device = torch.device('cpu')
    print('GPU is not available, using CPU instead')

# 创建张量并将其移到 GPU
x = torch.tensor([1, 2, 3]).to(device)

# 创建模型并将其移到 GPU
model = torch.nn.Linear(3, 1).to(device)

# 进行计算
output = model(x)
print(output)
