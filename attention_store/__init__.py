import abc
class AttentionStore :

    def __init__(self):

        self.query_dict = {}
        self.key_dict = {}
        self.attention_dict = {}
        self.query_list = []

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
    def save_attention(self, attention, layer_name):
        if layer_name not in self.attention_dict.keys():
            self.attention_dict[layer_name] = []
            self.attention_dict[layer_name].append(attention)

    def reset(self):

        self.query_dict = {}
        self.key_dict = {}
        self.attention_dict = {}
        self.query_list = []