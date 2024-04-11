import torch.nn as nn
import torch
import einops



class SinglePositionalEmbedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320,
                 pe_norm_type = None,
                 pe_dim = 3) :
        super(SinglePositionalEmbedding, self).__init__()
        self.pe_dim = pe_dim
        if pe_dim == 3:
            self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model),
                                                     requires_grad=True) # same shape with input
            self.norm = None
            if pe_norm_type is not None:
                self.norm_type = pe_norm_type
                if pe_norm_type == 'batch_norm':
                    self.norm = nn.BatchNorm1d(d_model)
                elif pe_norm_type == 'layer_norm':
                    self.norm = nn.LayerNorm(d_model)
                elif pe_norm_type == 'instance_norm':
                    self.norm = nn.InstanceNorm1d(d_model)
        elif pe_dim == 4:
            self.positional_encodings = nn.Parameter(torch.randn(1, int(max_len**0.5), int(max_len**0.5), d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):

        if self.pe_dim == 3:
            start_dim = 3
            if x.dim() == 4:
                start_dim = 4
                x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
            b_size = x.shape[0]
            res = int(x.shape[1] ** 0.5)
            pe = self.positional_encodings.expand(b_size, -1, -1).to(x.device)
            x = x + pe # dim = 3
            if start_dim == 4:
                x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)

            if self.norm is not None :
                self.norm = self.norm.to(x.device)
                if self.norm_type == 'batch_norm' or self.norm_type == 'instance_norm':
                    x = x.permute(0,2,1).contiguous()
                    x = self.norm(x)
                    x = x.permute(0,2,1).contiguous()
                else :
                    x= self.norm(x)
            return x
        elif self.pe_dim == 4:
            start_dim = 4
            if x.dim() == 3:
                start_dim = 3
                b,l,d = x.shape
                h = int(l ** 0.5)
                x = x.view(b,h,h,d)
            b_size = x.shape[0]
            pe = self.positional_encodings.expand(b_size, -1, -1, -1).to(x.device)
            x = x + pe
            if start_dim == 3:
                x = einops.rearrange(x, 'b h w c -> b (h w) c')
            return x


pe = SinglePositionalEmbedding(max_len = 32 * 32,
                               d_model = 320,
                               pe_norm_type = None)
pe_layer = pe.positional_encodings
b,l,d = pe_layer.shape
# print(b,l,d) # torch.Size([1, 1024, 320]) -> [batch, H,W, dim]
h = int(l ** 0.5)
pe_layer = pe_layer.view(b,h,h,d)
# print(pe_layer.shape) # torch.Size([1, 32, 32, 320]) -> [batch, 64,64, dim]

# new layer norm
N, C, H, W = 2, 3, 4, 4
input = torch.randn(N, C, H, W)
print(f'before = {input}')
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)
first = output[0]
print(f'after = {first.sum()}')