import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
import time

cudnn.benchmark = True

input = Variable(torch.randn(128, 3, 224, 224).cuda())
target = Variable(torch.LongTensor(128).fill_(1).cuda())
model = models.resnet50().cuda()
loss = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.001,
                            momentum=0.9,
                            weight_decay=1e-5)


for i in range(1000):
    torch.cuda.synchronize()
    start = time.time()
    out = model(input)
    err = loss(out, target)
    err.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(time.time() - start)
