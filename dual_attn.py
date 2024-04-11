import os
import torch
from torch import nn

A = torch.randn(1,320,64,64)
# convolution layer
b_conv = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
c_conv = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
B = b_conv(A)
C = c_conv(A)

# [2] reshaping (B,C,H,W) -> (B,C,H*W)
B = B.view(B.size(0), B.size(1), -1)
C = C.view(C.size(0), C.size(1), -1)
print(B.shape, C.shape)
# [3] matrix multiplication btw transpose of B and transpose of C and softmax
S = torch.matmul(B.transpose(1, 2), C)
# [4] softmax (shape : B, N, N) (almost self attn)
S = nn.functional.softmax(S, dim=-1) # if B component and C component are similar, the value is close to 1

# [5]
d_conv = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
D = d_conv(A)
# reshape to (B,C,N)
D = D.view(D.size(0), D.size(1), -1)

# [6] matrix multiplication between transpose of S and D
# permute to (B,N,C)
E = torch.matmul(D, S.transpose(1, 2)).permute(0,2,1)
# reshape to (B,C,H,W)

# -------------------------------------------------------------------------------------------------------------- #
# channel attention map (CXC)
A = torch.randn(1,320,64,64)
# [1] reshape ( -> (b,c,n))
b,c,h,w = A.shape
A = A.view(b,c,-1)
# matrix multiplication btw A and transpose of A and softmax
X = torch.matmul(A, A.transpose(1,2))
X = nn.functional.softmax(S, dim=-1) # channelwise importance
# matrix multiplication between X and transpose of A
Y = torch.matmul(X, A.transpose(1,2)).view(b,c,h,w)
print(Y.shape) # (1,320,4096) -> (b,c,n)


