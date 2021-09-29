from transformers.modeling_bert import BertForPreTraining
from xtools import *
from bert4torch.loss import CrossEntropyLoss
import random
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda import amp
from pytorch_lightning import seed_everything
import pkbar

# seed_everything(128)
batch_size = 32
model_path = 'hfl/chinese-roberta-wwm-ext'
# model_path = 'hfl/rbt3'
adam_epsilon = 1e-8
lr = 5e-5
device = 'cuda'
steps = 1000000
grad_accumulation_steps = 4
save_every = 100000

debug = 0


def get_data(path):
    ret = []
    _, name = os.path.split(path)

    name = name.split('_')[0]
    for idx, line in enumerate(tqdm.tqdm(open(path))):
        line = json.loads(line)
        line['task'] = name
        ret.append(line)
        if debug and idx > 10:
            break
    return ret


seq2seq_data = '/home/vocust001/xly/ccc/seq2seq_data.json'
lm_data = '/home/vocust001/xly/ccc/lm_data.json'
mlm_data = '/home/vocust001/xly/ccc/mlm_data.json'

print('start reading data')
seq2seq_data = get_data(seq2seq_data)
lm_data = get_data(lm_data)
mlm_data = get_data(mlm_data)

seq2seq_data = DataLoader(KeyDataset(seq2seq_data), batch_size=batch_size, collate_fn=default_collate, shuffle=True)
lm_data = DataLoader(KeyDataset(lm_data), batch_size=batch_size, collate_fn=default_collate, shuffle=True)
mlm_data = DataLoader(KeyDataset(mlm_data), batch_size=batch_size, collate_fn=default_collate, shuffle=True)

print('finish loading data')


def create_lm_mask(attention_mask, direction='l2r'):
    seq_len = attention_mask.size(-1)
    if attention_mask.ndim == 2:
        attention_mask = attention_mask.view(-1, 1, seq_len)

    idxs = torch.arange(0, seq_len).to(attention_mask)
    if direction == 'l2r':
        triu = (idxs.unsqueeze(-1) >= idxs).float()
    elif direction == 'r2l':
        triu = (idxs.unsqueeze(-1) <= idxs).float()

    attention_mask = (attention_mask + triu > 1).float()
    return attention_mask


def create_unilm_mask(s):
    idxs = torch.cumsum(s, axis=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = mask.float()
    return mask


class UnilmForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, *arg, **kwargs):
        prediction_scores, seq_relationship_score = super().forward(input_ids, attention_mask, token_type_ids)
        return prediction_scores, seq_relationship_score

    @classmethod
    def prepare_data_for_pretraining(self, batch, task, use_mlm=False):
        new_batch = batch
        if task == 'mlm':
            new_batch['input_ids'] = new_batch.pop('masked_input_ids')

        elif task == 'lm':
            new_batch['label_mask'] = new_batch.pop('attention_mask')
            if random.random() < 0.5:
                direction = 'l2r'
            else:
                direction = 'r2l'
            new_batch['attention_mask'] = create_lm_mask(new_batch['label_mask'], direction)
            new_batch['direction'] = direction

        elif task == 'seq2seq':
            new_batch['attention_mask'] = create_unilm_mask(new_batch.pop('attention_mask'))
        return new_batch

    @classmethod
    def compute_loss(self, logits, batch, task, direction=None):
        if task == 'mlm':
            mlm_logits, seq_logits = logits
            mlm_label = batch['mlm_labels']
            mask = mlm_label != 0
            mlm_loss = loss_fn(mlm_logits, mlm_label, mask)

            seq_label = batch['seq_label']
            seq_loss = loss_fn(seq_logits, seq_label, None)

            return mlm_loss + seq_loss

        elif task == 'seq2seq':
            logits, _ = logits
            label = batch['input_ids']
            logits = logits[:, :-1]
            label = label[:, 1:]
            loss = loss_fn(logits, label, batch['token_type_ids'][:, 1:])
            return loss

        elif task == 'lm':
            logits, _ = logits
            label = batch['input_ids']
            if direction == 'r2l':
                logits = logits[:, 1:]
                label = label[:, :-1]
                mask = batch['label_mask'][:, :-1]
            elif direction == 'l2r':
                logits = logits[:, :-1]
                label = label[:, 1:]
                mask = batch['label_mask'][:, 1:]
            loss = loss_fn(logits, label, mask)
            return loss


## 训练
model = UnilmForPreTraining.from_pretrained(model_path)
model.to(device)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=steps)

loss_fn = CrossEntropyLoss()
scaler = amp.GradScaler()


class BatchData:
    def __init__(self):
        self.mlm = mlm_data
        self.lm = lm_data
        self.seq2seq = seq2seq_data

        self.data = {'mlm': self.mlm._get_iterator(),
                     'lm': self.lm._get_iterator(),
                     'seq2seq': self.seq2seq._get_iterator()}

    def get_next_batch(self):
        if random.random() <= 0.5:
            batch_name = 'mlm'
        elif random.random() <= 0.5:
            batch_name = 'lm'
        else:
            batch_name = 'seq2seq'
        try:
            return next(self.data[batch_name])
        except Exception as e:
            self.data[batch_name] = self.init_new_iter(batch_name)
            return next(self.data[batch_name])

    def init_new_iter(self, name):
        return getattr(self, name)._get_iterator()


data = BatchData()
progress = pkbar.Kbar(target=steps, width=25)
print_loss = 0
for step in range(steps * grad_accumulation_steps):
    raw_batch = data.get_next_batch()
    batch = raw_batch.copy()
    task = batch.pop('task')[0]
    batch = UnilmForPreTraining.prepare_data_for_pretraining(batch, task)
    direction = batch.pop('direction') if 'direction' in batch else None
    batch = {k: v.to(device) for k, v in batch.items()}
    logtis = model(**batch)
    loss = UnilmForPreTraining.compute_loss(logtis, batch, task, direction)
    loss = loss / grad_accumulation_steps
    print_loss += loss.item()
    loss.backward()

    if (step + 1) % grad_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        progress.update(step, values=[('loss: ', round(print_loss, 4))])
        print_loss = 0

    if (step+1) % (grad_accumulation_steps * save_every) == 0:
        save_name = 'model_step_{}_loss_{}'.format(step, round(print_loss, 4))
        torch.save(model.state_dict(), save_name)
