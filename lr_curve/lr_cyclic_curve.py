import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import resnet18
import matplotlib.pyplot as plt

model = resnet18(num_classes=2)
base_lr=1e-4
max_lr=0.1
optimizer = optim.SGD(params=model.parameters(), lr=0.1)
scheduler = lr_scheduler.CyclicLR(optimizer,base_lr=base_lr,max_lr=max_lr,step_size_up=20,step_size_down=20)

#100 iteration
plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    # print(epoch, scheduler.get_lr()[0])    # get_lr()
    y.append(scheduler.get_lr()[0])
plt.plot(x, y)
plt.savefig('lr_cyclic.png')