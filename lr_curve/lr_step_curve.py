import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import resnet18
import matplotlib.pyplot as plt

model = resnet18(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=0.05)

# lr_scheduler.StepLR()
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    # print(epoch, scheduler.get_lr()[0])    # get_lr()
    y.append(scheduler.get_lr()[0])
plt.plot(x, y)
plt.savefig('lr_step.png')