from torch import nn
import torch
from attention_store import AttentionStore


def passing_argument(args):
    global do_local_self_attn
    global only_local_self_attn
    global fixed_window_size
    global argument

    argument = args


def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None, **model_kwargs):

            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            if noise_type is not None and argument.use_position_embedder :
                """ cross attention position embedding """
                position_model = noise_type[0]
                hidden_states = position_model(hidden_states, layer_name)

            query = self.to_q(hidden_states)
            # before saved query
            # query = [batch, pix_num, dim]
            # change query dim to 768

            if trg_layer_list is not None and layer_name in trg_layer_list:
                print(f'trg_layer_list: {trg_layer_list}')
                print(f"Layer Name: {layer_name} : query = {query.shape}")
                """
                if len(controller.query_list) == 0 :
                    controller.query_list.append(query)

                else :
                    before_query = controller.query_list[0]
                    controller.query_list = []
                    controller.query_list.append(query)
                    positioning = noise_type[1]
                    context = positioning(before_query, layer_name)
                """
            context = context if context is not None else hidden_states
            if type(context) == dict :
                p = query.shape[1]
                res = int(p ** 0.5)
                context = context[res]
            key_ = self.to_k(context)
            value = self.to_v(context)

            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key_)
            value = self.reshape_heads_to_batch_dim(value)
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            """ Second Trial """
            #if trg_layer_list is not None and layer_name in trg_layer_list :
                # batch=8, seq_len, dim
            #    controller.save_query((query * self.scale),layer_name) # query = batch, seq_len, dim

            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                                             query,
                                             key.transpose(-1, -2), beta=0, alpha=self.scale,) # [8, pix_num, sen_len]
            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
            #if trg_layer_list is not None and layer_name in trg_layer_list :
                # pix_num, pix_num
            #    controller.save_attention(attention_probs, layer_name) # attention_probs = batch, seq_len, sen_len

            hidden_states = torch.bmm(attention_probs, value) # [8, pix_num, dim]
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states) # 1, pix_num, dim

            if trg_layer_list is not None and layer_name in trg_layer_list:
                # [2] after channel attn [Batch, pix_num, dim]
                controller.save_query(hidden_states, layer_name)
            hidden_states = self.to_out[0](hidden_states)
            # it does not add original query again
            return hidden_states
        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count