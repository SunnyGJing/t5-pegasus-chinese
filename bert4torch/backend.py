from transformers import *
from bert4keras.tokenizers import Tokenizer
import torch
import torch.nn as nn


def forward_handler(func):
    def new_func(self, inputs):
        if not self.built:
            self.add_module('layer', self.layer_func(inputs.shape[-1], *self.args))
            self.layer.to(inputs.device)
            self.built = True
        return func(self, inputs)

    return new_func


def sequence_masking(x, mask, mode=0, axis=None):
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis < 0:
            axis = x.ndim + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = mask.unsqueeze(1)
        for _ in range(x.ndim - mask.ndim):
            mask = mask.unsqueeze(mask.ndim)
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


def mask_select(inputs, mask):
    input_dim = inputs.ndim
    mask_dim = mask.ndim
    mask = mask.reshape(-1).bool()

    if input_dim > mask_dim:
        inputs = inputs.reshape((int(mask.size(-1)), -1))[mask]
    else:
        inputs = inputs.reshape(-1)[mask]
    return inputs


def mask_loss(func):
    def new_func(self, inputs, targets, mask):
        if mask is not None:
            inputs = mask_select(inputs, mask)
            targets = mask_select(targets, mask)
        return func(self, inputs, targets)
    return new_func

def get_bert_model(model_path):
    model_paths = {
        'roberta': 'hfl/chinese-roberta-wwm-ext',
        'roberta-large': 'hfl/chinese-roberta-wwm-ext-large',
        'macbert': 'hfl/chinese-macbert-base',
        'macbert-large': 'hfl/chinese-macbert-large',
        'rbt3': 'hfl/rbt3'
    }
    if model_path in model_paths:
        model_path = model_paths[model_path]
    return BertModel.from_pretrained(model_path)


def simplify_vocab(dict_path, encoding='utf-8', startswith=None):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    new_token_dict, keep_tokens = {}, []
    startswith = startswith or []
    for t in startswith:
        new_token_dict[t] = len(new_token_dict)
        keep_tokens.append(token_dict[t])

    for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
        if t not in new_token_dict:
            keep = True
            if len(t) > 1:
                for c in Tokenizer.stem(t):
                    if (
                        Tokenizer._is_cjk_character(c) or
                        Tokenizer._is_punctuation(c)
                    ):
                        keep = False
                        break
            if keep:
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])

    return new_token_dict, keep_tokens


def shrink_embedding(bert_model, keep_tokens):
    vocab_size = bert_model.config.vocab_size
    keep_tokens = [x in keep_tokens for x in range(vocab_size)]
    emb_weight = bert_model.embeddings.word_embeddings.weight
    bert_model.embeddings.word_embeddings = nn.Parameter(emb_weight[keep_tokens])
    weight = bert_model.cls.predictions.decoder.weight
    bias = bert_model.cls.predictions.decoder.bias
    bert_model.cls.predictions.decoder.bias = bias[:, keep_tokens]
    bert_model.cls.predictions.decoder.weight = nn.Parameter(weight[keep_tokens])
    del emb_weight
    del weight
    del bias
