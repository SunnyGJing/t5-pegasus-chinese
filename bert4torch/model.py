from bert4torch.layer import *
from bert4torch.backend import *
from bert4torch.loss import CrossEntropyLoss


class FLATModel(nn.Module):
    def __init__(self, model_path, num_class, w2v_file, max_len, hidden_size=160, num_heads=8):
        super().__init__()
        self.embedding = FLATEmbedding(model_path, w2v_file)
        self.flat = FLAT(max_len=max_len, hidden_size=hidden_size, num_heads=num_heads)
        self.crf = LinearCRF(num_class)

    def forward(self, batch):
        a = batch['input_ids']
        b = batch['word_ids']
        c = batch['attention_mask']
        d = batch['word_mask']
        batch['char_word_vec'] = self.embedding(a, b, c, d)
        logits = self.flat(batch)
        logits = self.crf(logits)
        return logits

    def compute_loss(self, logits, batch):
        return self.crf.compute_loss(logits, batch['labels'], batch['attention_mask'])


class BiAffineNer(nn.Module):
    def __init__(self, model_path, num_class, loss_fn):
        super().__init__()
        self.bert = get_bert_model(model_path)
        self.biaffine = BiAffine(num_class)
        self.loss_fn = loss_fn

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        hidden = self.bert(input_ids, attention_mask)[0]
        logits = self.biaffine(hidden)
        return logits

    def compute_loss(self, logits, batch):
        masks = batch['attention_mask']
        logits_flat = []
        for logit, mask in zip(logits, masks):
            seq_len = mask.sum()
            indices_mat = torch.triu(torch.ones((seq_len, seq_len)), diagonal=0, out=None).to(mask.device)
            indices = torch.where(indices_mat > 0)
            logits_flat.append(logit[indices[0], indices[1]])
        logits_flat = torch.cat(logits_flat, 0)
        loss = self.loss_fn(logits_flat, batch['labels'], mask=None)
        return loss


class UniLM(BertForMaskedLM):
    def __init__(self, config, loss_fn=CrossEntropyLoss()):
        super().__init__(config)
        self.loss_fn = loss_fn

    def create_mask(self, batch):
        token_type_ids = batch['token_type_ids']
        idxs = torch.cumsum(token_type_ids, dim=-1)
        mask = idxs[:, None, :] <= idxs[:, :, None].to(batch['token_type_ids'])
        return mask

    def compute_loss(self, logits, batch):
        targets = batch['input_ids'][:, 1:]
        mask = batch['token_type_ids'][:, 1:]
        logits = logits[:, :-1]
        loss = self.loss_fn(logits, targets, mask)
        return loss

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, token_type_ids=None,
                                      **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        if token_type_ids is not None:
            delta = input_shape[1] - token_type_ids.shape[1]
            if delta > 0:
                new_type_id = torch.ones((input_shape[0], delta), device=input_ids.device, dtype=torch.long)
                token_type_ids = torch.cat([token_type_ids, new_type_id], dim=-1)
        return {"input_ids": input_ids, 'attention_mask': attention_mask,
                "token_type_ids": token_type_ids}


class ResidualGatedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dense0 = Dense(384)
        self.residuals = [ResidualGatedConv1D(3, 1),
                          ResidualGatedConv1D(3, 2),
                          ResidualGatedConv1D(3, 4),
                          ResidualGatedConv1D(3, 8),
                          ResidualGatedConv1D(3, 1),
                          ResidualGatedConv1D(3, 1)]
        self.dense = Dense(1, activation='sigmoid')

    def forward(self, logtis, mask=None):
        if mask is not None:
            logtis = sequence_masking(logtis, mask)
        logtis = self.dropout(logtis)
        logtis = self.dense0(logtis)
        logtis = self.dropout(logtis)
        for layer in self.residuals:
            logtis = layer(logtis, mask)
            logtis = self.dropout(logtis)
        prob = self.dense(logtis)
        return prob

    def compute_loss(self, prob, label):
        loss_fn = nn.BCELoss()
        return loss_fn(prob, label)

# from transformers import RobertaForCausalLM
# from transformers.modeling_roberta import RobertaPreTrainedModel
# from transformers.modeling_bert import BertModel
#
# BertModel.generate
# RobertaPreTrainedModel.generate
# RobertaForCausalLM.from_pretrained()
# RobertaForCausalLM.forward
# RobertaForCausalLM.generate
# AutoModelForSeq2SeqLM.from_pretrained()
# # from transformers.models.encoder_decoder import
#
# BertForMaskedLM.forward
# RobertaForCausalLM.from_pretrained()

