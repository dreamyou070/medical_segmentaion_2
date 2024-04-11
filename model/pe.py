import torch.nn as nn
import torch
import einops
"""
class ContextEmbedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320,
                 scale = 1.0,) :
        super(ContextEmbedding, self).__init__()
        # three convolution
        self.b_conv = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.c_conv = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.d_conv = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.scale = scale

    def forward(self, x: torch.Tensor):

        start_dim = 4
        if x.dim() == 3:
            start_dim = 3
            # x = B,L,D -> B,D,H,W
            b,l,d = x.shape
            h = int(l ** 0.5)
            # B,L,D -> B,D,L
            x = x.permute(0,2,1).contiguous()
            x = x.view(b,d,h,h)

        self.b_conv = self.b_conv.to(x.device).to(x.dtype)
        self.c_conv = self.c_conv.to(x.device).to(x.dtype)
        self.d_conv = self.d_conv.to(x.device).to(x.dtype)

        B = self.b_conv(x)
        C = self.c_conv(x)
        B = B.view(B.size(0), B.size(1), -1)
        C = C.view(C.size(0), C.size(1), -1)
        S = torch.matmul(B.transpose(1, 2), C)
        S = nn.functional.softmax(S, dim=-1)
        # [5]
        D = self.d_conv(x)
        D = D.view(D.size(0), D.size(1), -1)
        context_info = torch.matmul(D, S.transpose(1, 2)).permute(0,2,1) * self.scale
        return context_info
"""
class SinglePositionalEmbedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320) :
        super(SinglePositionalEmbedding, self).__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)


    def forward(self, x: torch.Tensor):

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
        return x

class AllPositionalEmbedding(nn.Module):

    layer_names_res_dim = {'down_blocks_0_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'down_blocks_0_attentions_1_transformer_blocks_0_attn2': (64, 320),

                           'down_blocks_1_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'down_blocks_1_attentions_1_transformer_blocks_0_attn2': (32, 640),

                           'down_blocks_2_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'down_blocks_2_attentions_1_transformer_blocks_0_attn2': (16, 1280),

                           'mid_block_attentions_0_transformer_blocks_0_attn2': (8, 1280),

                           'up_blocks_1_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_1_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_2_transformer_blocks_0_attn2': (16, 1280),

                           'up_blocks_2_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_1_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_2_transformer_blocks_0_attn2': (32, 640),

                           'up_blocks_3_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_1_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_2_transformer_blocks_0_attn2': (64, 320), }

    def __init__(self, use_context_info = True,
                 context_scale = 1.0) -> None:
        super(AllPositionalEmbedding, self).__init__()

        self.layer_dict = self.layer_names_res_dim
        self.positional_encodings = {}
        self.use_context_info = use_context_info
        self.context_encodings = {}
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            self.positional_encodings[layer_name] = SinglePositionalEmbedding(max_len = res*res,
                                                                              d_model = dim,)
            """
            if self.use_context_info :
                self.context_encodings[layer_name] = ContextEmbedding(max_len=res * res,
                                                                      d_model=dim,
                                                                      scale=context_scale)
            """

    def forward(self, x: torch.Tensor, layer_name):
        if layer_name in self.positional_encodings.keys() :
            position_embedder = self.positional_encodings[layer_name]
            output = position_embedder(x)
            #if self.use_context_info:
            #    context_embedder = self.context_encodings[layer_name]
            #    context_info = context_embedder(x)
            #    output = output + context_info
            return output
        else :
            return x