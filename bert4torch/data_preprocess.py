from transformers import BertTokenizerFast
import collections
import six
import torch
import os
import re
from utils import *

tokenizer = BertTokenizerFast.from_pretrained('hfl/rbt3')


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self, w_list=None):
        self.root = TrieNode()
        if w_list is not None:
            for w in w_list:
                self.insert(w)

    def insert(self, w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self, w):
        '''
        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)
            if current is None:
                return -1
        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break
                if current.is_w:
                    result.append([i, j, sentence[i:j + 1]])
        return result


class FLATDataProcessor(object):
    def __init__(self, id2label_or_labels, vocab):
        self.trie = Trie(vocab)
        self.vocab2id = dict(zip(vocab, range(len(vocab))))
        self.init_label(id2label_or_labels)

    def init_label(self, id2label_or_labels):
        if isinstance(id2label_or_labels[0], list):
            self.id2label = set()
            for line in id2label_or_labels:
                for item in line:
                    self.id2label.add(item['label'])
        elif isinstance(id2label_or_labels, list):
            self.id2label = id2label_or_labels
        self.id2label = sorted(self.id2label)
        print('Unique labels: {}'.format(self.id2label))
        self.label2id = dict(zip(self.id2label, range(len(self.id2label))))

    @property
    def num_class(self):
        return 2 * (len(self.label2id)) + 1

    def encode(self, text, label=None):
        features = tokenizer.encode_plus(text, return_offsets_mapping=True, return_token_type_ids=False,
                                         max_length=512)

        mapping = {}
        for idx, (s, e) in enumerate(features['offset_mapping'][1:-1]):
            for i in range(s, e):
                mapping[i] = idx + 1

        char_len = len(features['input_ids'])
        starts = list(range(char_len))
        ends = starts[:]
        word_mask = [0] * char_len
        word_ids = word_mask[:]

        lattice = self.trie.get_lexicon(text)
        for (start, end, word) in lattice:
            new_start = mapping.get(start)
            new_end = mapping.get(end)
            if new_start is None or new_end is None:
                print('vocab {} is ignored.'.format(word))
                continue
            starts.append(new_start)
            ends.append(new_end)
            word_mask.append(1)
            word_ids.append(self.vocab2id[word])
        char_word_mask = [1] * len(word_mask)
        offset = features.pop('offset_mapping')
        features.update({'start': starts, 'end': ends, 'char_word_mask': char_word_mask,
                         'word_mask': word_mask, 'word_ids': word_ids})
        if label is not None:
            labels = [0] * char_len
            for item in label:
                label_id = self.label2id[item['label']]
                spans = item.get('spans')
                print(spans)
                if spans is None:
                    starts = search(item['entity'], text)
                    if starts == -1:
                        continue
                    spans = ((start, start + len(item['entity']) - 1) for start in starts)

                for start, end in spans:
                    new_start = mapping[start]
                    new_end = mapping[end]
                    # print(start_mapping, end_mapping)

                    labels[new_start] = 2 * label_id + 1
                    for i in range(new_start + 1, new_end + 1):
                        labels[i] = 2 * (label_id + 1)
            features.update({'labels': labels, })
        return features, offset

    def batch_encode(self, texts, labels, num_works=1):
        if labels is None:
            labels = [None] * len(texts)

        if num_works == 1:
            return list(zip(*[self.encode(*inputs) for inputs in zip(texts, labels)]))
        else:
            from bert4keras.snippets import parallel_apply
            return parallel_apply(self.encode, zip(texts, labels), workers=num_works, max_queue_size=5000)


class NERDataProcessor(object):
    def __init__(self, id2label_or_labels):

        self.init_label(id2label_or_labels)

    def init_label(self, id2label_or_labels):
        if isinstance(id2label_or_labels[0], list):
            self.id2label = set()
            for line in id2label_or_labels:
                for item in line:
                    self.id2label.add(item['label'])
        elif isinstance(id2label_or_labels, list):
            self.id2label = id2label_or_labels
        self.id2label = sorted(self.id2label)
        print('Unique labels: {}'.format(self.id2label))
        self.label2id = dict(zip(self.id2label, range(len(self.id2label))))

    @property
    def num_class(self):
        return 2 * (len(self.label2id)) + 1

    def encode(self, text, label=None):

        features = tokenizer.encode_plus(text, return_offsets_mapping=True, return_token_type_ids=False,
                                         max_length=512)

        mapping = {}
        for idx, (s, e) in enumerate(features['offset_mapping'][1:-1]):
            for i in range(s, e):
                mapping[i] = idx + 1

        offset = features.pop('offset_mapping')
        char_len = len(offset)

        if label is not None:
            labels = [0] * char_len
            for item in label:
                label_id = self.label2id[item['label']]
                spans = item.get('spans')
                if spans is None:
                    starts = search(item['entity'], text)
                    if starts == -1:
                        continue
                    spans = ((start, start + len(item['entity']) - 1) for start in starts)

                for start, end in spans:
                    new_start = mapping[start]
                    new_end = mapping[end]
                    # print(start_mapping, end_mapping)

                    labels[new_start] = 2 * label_id + 1
                    for i in range(new_start + 1, new_end + 1):
                        labels[i] = 2 * (label_id + 1)

            features.update({'labels': labels, })
        return features, offset

    def batch_encode(self, texts, labels, num_works=1):
        if labels is None:
            labels = [None] * len(texts)

        if num_works == 1:
            return list(zip(*[self.encode(*inputs) for inputs in zip(texts, labels)]))
        else:
            from bert4keras.snippets import parallel_apply
            return parallel_apply(self.encode, zip(texts, labels), workers=num_works, max_queue_size=5000)


class BiaffineDataProcessor(object):
    def __init__(self, id2label_or_labels):

        self.init_label(id2label_or_labels)

    def init_label(self, id2label_or_labels):
        if isinstance(id2label_or_labels[0], list):
            self.id2label = set()
            for line in id2label_or_labels:
                for item in line:
                    self.id2label.add(item['label'])
        elif isinstance(id2label_or_labels, list):
            self.id2label = id2label_or_labels
        self.id2label = sorted(self.id2label)
        print('Unique labels: {}'.format(self.id2label))
        self.label2id = dict(zip(self.id2label, range(len(self.id2label))))

    @property
    def num_class(self):
        return len(self.label2id) + 1

    def encode(self, text, label=None):

        features = tokenizer.encode_plus(text, return_offsets_mapping=True, return_token_type_ids=False,
                                         max_length=512)

        mapping = {}
        for idx, (s, e) in enumerate(features['offset_mapping'][1:-1]):
            for i in range(s, e):
                mapping[i] = idx + 1

        offset = features.pop('offset_mapping')
        char_len = len(offset)

        if label is not None:
            labels = [0] * ((char_len + 1) * char_len // 2)
            for item in label:
                label_id = self.label2id[item['label']]
                spans = item.get('spans')
                if spans is None:
                    starts = search(item['entity'], text)
                    if starts == -1:
                        continue
                    spans = ((start, start + len(item['entity']) - 1) for start in starts)
                label_mapping = {}
                for i in range(char_len):
                    for j in range(i, char_len):
                        label_mapping[(i, j)] = len(label_mapping)

                for start, end in spans:
                    new_start = mapping[start]
                    new_end = mapping[end]
                    labels[label_mapping[(new_start, new_end)]] = label_id + 1
            features.update({'labels': labels, })
        return features, offset

    def batch_encode(self, texts, labels, num_works=1):
        if labels is None:
            labels = [None] * len(texts)

        if num_works == 1:
            return list(zip(*[self.encode(*inputs) for inputs in zip(texts, labels)]))
        else:
            from bert4keras.snippets import parallel_apply
            return parallel_apply(self.encode, zip(texts, labels), workers=num_works, max_queue_size=5000)


def create_flat_dataloader(texts, labels, processor, batch_size, shuffle=False, num_works=1, cache=False):
    if cache and os.path.exists(cache):
        print('Recovering data from {}'.format(cache))
        return torch.load(cache)
    features = processor.batch_encode(texts, labels, num_works)
    dataloader = create_dataloader(features, batch_size, shuffle, cache)
    return dataloader


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    ret = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            ret.append(i)
    return ret or -1


# if __name__ == "__main__":
    # vocab = ['我们', '我们是', '我们是冠军', '是冠军', '冠军']
    # text = 'hello laughing 我们是冠军'
    # labels = ['a', 'b']
    # label = {'我们': 'a', '冠军': 'b'}
    # flat = FLATDataProcessor(labels, vocab)
    # for k, v in flat.encode(text, label).items():
    #     print(k, v)

    import tqdm
