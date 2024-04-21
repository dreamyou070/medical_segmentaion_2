import torch.nn as nn
import torch
import einops

class SinglePositional_Patch_Embedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1).to(x.device)
        self.linear = self.linear.to(x.device)
        pe = self.linear(pe)
        x = x + pe
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x
class SinglePositionalEmbedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1).to(x.device)
        x = x + pe
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x
class SinglePositional_Semantic_Embedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320,):
        super().__init__()
        # [1] positional embeddings
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)
        # [2] semantic embeddings
        self.se_alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, x: torch.Tensor):

        # [1] positional embeddings
        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1).to(x.device)
        # [2] semantic embeddings
        attention_scores = torch.baddbmm(torch.empty(x.shape[0], x.shape[1], x.shape[1], dtype=x.dtype, device=x.device),
                                         x, x.transpose(-1, -2),beta=0,)
        attention_probs = attention_scores.softmax(dim=-1).to(x.dtype)
        se = torch.bmm(attention_probs, x)
        # [3] add positional and semantic embeddings
        x = x + pe + self.se_alpha.to(x.device) * se
        # [4] reshape
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x
class SinglePositionalEmbedding_concat(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)
        # [2] dimension reduction
        self.fc = nn.Linear(2*d_model, d_model)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1).to(x.device)
        # [1] concat query and position_embedder
        x = torch.cat([x, pe], dim=2)
        # [2] reshape query (dimension reduction)
        self.fc = self.fc.to(x.device)
        x = self.fc(x)
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

    def __init__(self,
                 pe_do_concat,
                 do_semantic_position,) -> None:
        super().__init__()

        self.layer_dict = self.layer_names_res_dim
        self.positional_encodings = {}
        self.do_concat = pe_do_concat
        self.do_semantic_position = do_semantic_position
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]

            if self.do_concat :
                self.positional_encodings[layer_name] = SinglePositionalEmbedding_concat(max_len = res*res, d_model = dim)

            elif self.do_semantic_position :
                self.positional_encodings[layer_name] = SinglePositional_Semantic_Embedding(max_len = res*res, d_model = dim)

            else :
                self.positional_encodings[layer_name] = SinglePositionalEmbedding(max_len = res*res, d_model = dim)



    def forward(self,
                x: torch.Tensor,
                layer_name,
                patch_idx=None):


        if layer_name in self.positional_encodings.keys() :

            position_embedder = self.positional_encodings[layer_name]
            output = position_embedder(x)

            return output
        else :
            return x


class SingleInternalCrossAttention(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.layer = nn.Linear(2*d_model, 768)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        self.fc = self.fc.to(x.device)
        x = self.fc(x)
        return x

class AllInternalCrossAttention(nn.Module):

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

    def __init__(self,
                 ) -> None:
        super().__init__()

        self.layer_dict = self.layer_names_res_dim
        self.internal_cross_encodings = {}
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            self.internal_cross_encodings[layer_name] = SingleInternalCrossAttention(max_len = res*res,
                                                                                     d_model = dim)

    def forward(self,
                x: torch.Tensor,
                layer_name,
                patch_idx=None):

        if layer_name in self.internal_cross_encodings.keys() :
            internal_layer = self.internal_cross_encodings[layer_name]
            output = internal_layer(x)
            return output

        else :
            return x