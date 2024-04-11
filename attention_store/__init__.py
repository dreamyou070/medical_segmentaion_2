import abc
class AttentionStore :

    def __init__(self):

        self.query_dict = {}
        self.key_dict = {}

    def get_empty_store(self):
        return {}

    def save_query(self, query, layer_name):

        if layer_name not in self.query_dict.keys():
            self.query_dict[layer_name] = []
            self.query_dict[layer_name].append(query)
        else:
            self.query_dict[layer_name].append(query)

    def save_key(self, key, layer_name) :

        if layer_name not in self.key_dict.keys():
            self.key_dict[layer_name] = []
            self.key_dict[layer_name].append(key)
        else:
            self.key_dict[layer_name].append(key)


    def reset(self):

        self.query_dict = {}
        self.key_dict = {}

# layer_name : down_blocks_0_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_0_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_0_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_0_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_1_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_1_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_1_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_1_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_2_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : down_blocks_2_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : down_blocks_2_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : down_blocks_2_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : mid_block_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : mid_block_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_2_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_2_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_2_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_2_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_2_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_3_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_2_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_2_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897