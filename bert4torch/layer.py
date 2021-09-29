import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import math
from bert4torch.backend import *
from torch.nn.modules.utils import _single, _pair, _triple


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.built = False

    def build(self, inputs):
        self.built = True


class LayerNorm(Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.layer_func = nn.LayerNorm
        self.built = False

    @forward_handler
    def forward(self, inputs):
        return self.layer(inputs)


class Dense(Module):
    def __init__(self, *args, activation=None):
        super().__init__()
        self.args = args
        self.get_activation(activation)
        self.layer_func = nn.Linear
        self.built = False

    @forward_handler
    def forward(self, inputs):
        if self.activation is not None:
            return self.activation(self.layer(inputs))
        return self.layer(inputs)

    def get_activation(self, name):
        if isinstance(name, str) and hasattr(F, name):
            self.activation = getattr(F, name)
        elif name is None:
            self.activation = None
        else:
            raise ValueError('activation not found')


class W2VEmbedding(nn.Module):
    def __init__(self, pretrain_file=None, embedding_size=None, vocab_size=None, weights=None):
        super().__init__()
        self.pretrain_file = pretrain_file
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.weights = weights
        if self.pretrain_file is not None:
            self.load_pretrain()
        self.encoder = nn.Embedding(self.vocab_size, self.embedding_size, _weight=self.weights)

    def load_pretrain(self):
        embeddings = []
        for idx, line in enumerate(open(self.pretrain_file, encoding='utf-8')):
            vocab, *vector = line.strip().split()
            embeddings.append(np.array(vector).astype('float32'))
        embeddings = np.vstack(embeddings)
        self.weights = torch.from_numpy(embeddings)
        self.vocab_size, self.embedding_size = embeddings.shape
        print('Embeddings shape: {}'.format(embeddings.shape))
        return embeddings

    def forward(self, input_ids):
        return self.encoder(input_ids)


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        mask = mask.bool()
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator
        llh = - llh
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()

        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        mask = mask.bool()
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        mask = mask.bool()
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class LinearCRF(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.dense = Dense(num_class)
        self.crf = CRF(num_class)

    def forward(self, hidden):
        return self.dense(hidden)

    def compute_loss(self, logits, targets, mask):
        return self.crf(logits, targets, mask)


# module for flat
class FLATConfig(object):
    def __init__(self):
        self.num_flat_layers = 1
        self.num_pos = 4
        self.dim_pos = 160
        self.hidden_size = 160  # out_size
        self.num_heads = 8
        self.scaled = False
        self.attn_dropout = 0.1
        self.hidden_dropout = 0.1
        self.en_ffd = True
        self.layer_norm_eps = 1e-12
        self.intermediate_size = 640
        self.dropout = 0.1
        self.in_feat_size = 160
        self.out_feat_size = self.hidden_size
        self.max_len = 600


class RelPositionEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super(RelPositionEmbedding, self).__init__()
        self.max_len = max_len
        num_embedding = max_len * 2 - 1
        half_dim = int(dim // 2)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-max_len + 1, max_len, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embedding, -1)
        if dim % 2 == 1:
            print('embedding dim is odd')
            emb = torch.cat([emb, torch.zeros(num_embedding, 1)], dim=1)
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.dim = dim

    def forward(self, pos):
        pos = pos + (self.max_len - 1)
        pos_shape = pos.size()
        pos_emb = self.emb[pos.view(-1)]
        pos_emb = pos_emb.reshape(list(pos_shape) + [self.dim])
        return pos_emb


class MultiHeadAttentionRel(Module):
    def __init__(self, num_heads, scaled=True, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scaled = scaled
        self.dropout = nn.Dropout(attn_dropout)
        self.per_head_size = None

    def build(self, inputs):
        input_size = inputs.size(-1)
        self.built = True
        self.per_head_size = input_size // self.num_heads

        self.add_module('w_k', Dense(input_size))
        self.add_module('w_q', Dense(input_size))
        self.add_module('w_v', Dense(input_size))
        self.add_module('w_r', Dense(input_size))

        u = nn.init.xavier_normal_(
            torch.randn(self.num_heads, self.per_head_size, device=inputs.device)
        )
        v = nn.init.xavier_normal_(
            torch.randn(self.num_heads, self.per_head_size, device=inputs.device))

        self.register_parameter('u', nn.Parameter(u, requires_grad=True))
        self.register_parameter('v', nn.Parameter(v, requires_grad=True))
        self.u.to(inputs.device)
        self.v.to(inputs.device)

    def forward(self, key, query, value, pos, key_mask):
        if not self.built:
            self.build(key)

        key = self.w_k(key)
        query = self.w_q(query)
        value = self.w_v(value)
        rel_pos_embedding = self.w_r(pos)

        batch, _, hidden_size = key.size()

        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, -1, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, -1, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, -1, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          list(rel_pos_embedding.size()[:3]) + [self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)

        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        query_and_u_for_c = query + u_for_c
        A_C = torch.matmul(query_and_u_for_c, key)

        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        query_for_b = query.view([batch, self.num_heads, query.size(2), 1, self.per_head_size])
        query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)

        attn_score_raw = A_C + B_D

        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        mask = key_mask.unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw - (1 - mask) * 1e15
        attn_score = F.softmax(attn_score_raw_masked, dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, -1, hidden_size)

        return result


class FeedForward(Module):
    def __init__(self, intermediate_size=1024):
        super().__init__()
        self.intermediate_size = intermediate_size

    def build(self, inputs):
        self.built = True

        input_size = inputs.size(-1)
        self.add_module('dense', nn.Sequential(
            Dense(self.intermediate_size, activation='gelu'),
            Dense(input_size)
        ))
        self.dense.to(inputs.device)
        # self.add_module('dense1', Dense(self.intermediate_size, activation='gelu'))
        # self.add_module('dense2', Dense(input_size))

    def forward(self, x):
        if not self.built:
            self.build(x)
        return self.dense(x)
        # return self.dense2(self.dense1(x))


class RelTransformerEncoderLayer(Module):
    def __init__(self, num_heads, scaled=True, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttentionRel(num_heads, scaled, dropout)
        self.ff = FeedForward()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def build(self, inputs):
        self.built = True
        input_size = inputs.size(-1)
        self.add_module('layer_norm1', nn.LayerNorm(input_size, eps=1e-6))
        self.add_module('layer_norm2', nn.LayerNorm(input_size, eps=1e-6))

        self.layer_norm1.to(inputs.device)
        self.layer_norm2.to(inputs.device)

    def forward(self, hidden, mask, pos_embedding):
        if not self.built:
            self.build(hidden)
        attn_out = self.attn(hidden, hidden, hidden, pos_embedding, mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layer_norm1(hidden + attn_out)
        ff_out = self.ff(out1)
        ff_out = self.dropout2(ff_out)
        out2 = self.layer_norm2(out1 + ff_out)
        return out2


class FLATEmbedding(Module):
    def __init__(self, model_path, w2v_file, dropout=0.1):
        super(FLATEmbedding, self).__init__()
        self.bert = get_bert_model(model_path)
        self.w2v = W2VEmbedding(w2v_file)
        self.w2v_linear = Dense(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size, eps=1e-6)

    def forward(self, char_ids, word_ids, char_mask, word_mask):
        char_vec = self.bert(char_ids, char_mask)[0] * char_mask[..., None]
        word_vec = self.w2v(word_ids) * word_mask[..., None]
        word_vec = self.w2v_linear(word_vec) * word_mask[..., None]

        word_vec = self.dropout(word_vec)
        word_vec = self.layer_norm(word_vec)

        batch_size, word_len, embedding_size = word_vec.size()

        char_vec = torch.cat(
            [char_vec, torch.zeros((batch_size, word_len - char_vec.size(1), embedding_size)).to(char_vec)],
            dim=1)
        char_word_vec = char_vec + word_vec
        return char_word_vec


class FLAT(Module):
    def __init__(self, max_len, hidden_size, num_heads, num_layers=1, scaled=True, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.pe = RelPositionEmbedding(max_len, hidden_size)
        self.adapter = Dense(hidden_size)
        self.pos_dense = Dense(hidden_size, activation='relu')
        self.encoder_layers = []
        for _ in range(num_layers):
            encoder_layer = RelTransformerEncoderLayer(num_heads, scaled=scaled, dropout=dropout)
            self.encoder_layers.append(encoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

    def forward(self, inputs):
        char_word_vec = inputs['char_word_vec']
        char_word_mask = inputs['char_word_mask']
        char_word_s = inputs['start']
        char_word_e = inputs['end']
        char_mask = inputs['attention_mask']
        max_len = char_mask.sum(1).max()

        hidden = self.adapter(char_word_vec)

        pe_ss = self.pe(char_word_s.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_se = self.pe(char_word_s.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))
        pe_es = self.pe(char_word_e.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_ee = self.pe(char_word_e.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))

        pos_embedding = self.pos_dense(torch.cat([pe_ss, pe_se, pe_es, pe_ee], -1))

        for layer in self.encoder_layers:
            hidden = layer(hidden, char_word_mask, pos_embedding)
        # char_vec = hidden[:, :max_len] * char_mask[..., None]
        char_vec = self.compute_mask(hidden[:, :max_len], char_mask, 'mul')
        return char_vec


class LayerNormFeedForward(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm()
        self.ff = FeedForward(1024)

    def forward(self, hidden):
        out1 = self.ff(hidden)
        out2 = self.layer_norm(out1 + hidden)
        return out2


class BiAffine(Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.start_ff = LayerNormFeedForward()
        self.end_ff = LayerNormFeedForward()

    def build(self, inputs):
        self.built = True
        input_size = inputs.size(-1)
        u = nn.init.xavier_normal_(
            torch.randn(input_size, self.num_class, input_size, device=inputs.device)
        )
        self.register_parameter('u', nn.Parameter(u, requires_grad=True))

    def forward(self, logits):
        if not self.built:
            self.build(logits)
        start_logits = self.start_ff(logits)
        end_logits = self.end_ff(logits)
        start_end_logits = torch.einsum('bik,knd,bjd->bijn', start_logits, self.u, end_logits)
        return start_end_logits


class MRC(nn.Module):
    def __init__(self, num_class=2, loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.num_class = num_class
        self.start_ff = LayerNormFeedForward()
        self.end_ff = LayerNormFeedForward()
        self.start_linear = Dense(num_class)
        self.end_linear = Dense(num_class)
        self.loss_fn = loss_fn

    def forward(self, logits):
        start_logits = self.start_ff(logits)
        start_logits = self.start_linear(start_logits)
        end_logits = self.end_ff(logits)
        end_logits = self.end_linear(end_logits)
        return start_logits, end_logits

    def compute_loss(self, logits, batch):
        masks = batch['label_mask']
        start_logits, end_logits = logits
        start_loss = self.loss_fn(start_logits, batch['start_label'], masks)
        end_loss = self.loss_fn(end_logits, batch['end_label'], masks)
        loss = start_loss + end_loss
        return loss


class ResidualGatedConv1D(Module):
    """门控卷积
    """

    def __init__(self, kernel_size, dilation_rate=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.layernorm = LayerNorm()

    def build(self, inputs):
        super().build(inputs)
        self.filters = inputs.size(-1)
        conv1d = Conv1d(inputs.size(-1),
                        inputs.size(-1) * 2,
                        kernel_size=self.kernel_size,
                        dilation=self.dilation_rate, )
        self.add_module('conv1d', conv1d)
        self.conv1d.to(inputs.device)
        self.register_parameter('alpha', nn.Parameter(torch.zeros((1,), device=inputs.device), requires_grad=True))

    def forward(self, logits, mask=None):
        if not self.built:
            self.build(logits)

        if mask is not None:
            logits = sequence_masking(logits, mask)

        outputs = self.conv1d(logits.transpose(1, 2))
        outputs = outputs.transpose(1, 2)
        gate = outputs[..., self.filters:].contiguous().sigmoid()
        outputs = (outputs[..., :self.filters] * gate).contiguous()
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            logits = self.dense(logits)
        # outputs = logits + self.alpha * outputs
        outputs = logits + outputs
        return outputs


class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def conv1d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    # padding_cols = max(0, (out_rows - 1) * stride[0] +
    #                     (filter_rows - 1) * dilation[0] + 1 - input_rows)
    # cols_odd = (padding_rows % 2 != 0)

    # if rows_odd or cols_odd:
    #     input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    if rows_odd:
        input = F.pad(input, [0, 0, int(rows_odd)])
    return F.conv1d(input, weight, bias, stride,
                    padding=padding_rows // 2,
                    dilation=dilation, groups=groups)


class Conv1d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


if __name__ == "__main__":
    flat = FLAT(max_len=20, hidden_size=30, num_heads=2)
    inputs = {
        'char_word_vec': torch.randn(2, 4, 100),
        'char_word_mask': torch.tensor([[1, 1, 1, 0],
                                        [1, 1, 0, 0]]),
        'start': torch.tensor([[0, 1, 0, 0],
                               [0, 1, 1, 0]]),
        'end': torch.tensor([[0, 1, 2, 0],
                             [0, 1, 1, 0]]),
        'attention_mask': torch.tensor([[1, 1],
                                        [1, 0]]),
    }
    print(flat(inputs))
