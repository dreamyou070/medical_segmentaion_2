import torch.nn as nn
import torch
import einops

class SingleInternalCrossAttention(nn.Module):

    def __init__(self,
                 d_model: int = 320, ):
        super().__init__()
        self.layer = nn.Linear(d_model, 768)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C # batch, len, dim
            x = x.permute(0,2,1).contiguous()
        x = self.layer.to(x.device)(x)                                    # batch, len, dim
        return x


class SelfFeatureMerger(nn.Module):

    # all cross attention #

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

    def __init__(self,) -> None:
        super().__init__()

        self.layer_dict = self.layer_names_res_dim
        self.internal_cross_encodings = {}
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            self.internal_cross_encodings[layer_name] = SingleInternalCrossAttention(d_model = dim)
            print(f'Layer {layer_name} is added to the internal cross attention list')

    def forward(self,
                x: torch.Tensor,
                layer_name):

        if layer_name in self.internal_cross_encodings.keys() :
            output = self.internal_cross_encodings[layer_name](x)
            return output

        else :
            return x

