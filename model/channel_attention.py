from torch import nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # aveage pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # no weight (just pooling)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # channel change
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # sum of averaging pooling and max pooling
        # [1] avg pool out mean the average value
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # [2] max pool out mean the maximum value
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # [3] final is sum of two values
        out = avg_out + max_out

        # non linear mapping
        return self.sigmoid(out)
import torch
# avg_pool = spatial dimension averaging
# max_pool = spatial dimension max num
input = torch.randn(1,4,64,64)
avg_pool = nn.AdaptiveAvgPool2d(1)
print(avg_pool.weight)
#output = avg_pool(input)
#print(output.shape)

#max_pool = nn.AdaptiveMaxPool2d(1)
#output_2 = max_pool(input)
#print(output_2.shape)