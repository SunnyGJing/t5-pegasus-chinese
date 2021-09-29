import collections
import torch


def ner_span_f1(y_pred, y_true, ignore_spans=True):
    if ignore_spans:
        y_pred = [{(x['entity'], x['label']) for x in y} for y in y_pred]
        y_true = [{(x['entity'], x['label']) for x in y} for y in y_true]

    # label = {x['label'] for y in y_true for x in y}

    tp = fp = fn = 0
    for y_p, y_t in zip(y_pred, y_true):
        tp_pair = y_p & y_t
        fp_pair = y_p - tp_pair
        fn_pair = y_t - tp_pair
        tp += len(tp_pair)
        fp += len(fp_pair)
        fn += len(fn_pair)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return round(f1, 4)


def predict_biaffine(model, data_loader, texts, offset, id2label):
    model.eval()
    y_pred = []
    for batch in data_loader:
        with torch.no_grad():
            logits = model(batch)
        for l in logits:
            y_pred.append(l)
    y_pred = extract_biaffine(y_pred, texts, offset, id2label)
    return y_pred


def predict_classical_ner(model, data_loader, texts, offset, id2label):
    model.eval()
    y_pred = []
    for batch in data_loader:
        with torch.no_grad():
            logits = model(batch)
        pred = model.net.crf.crf.decode(logits, batch['attention_mask'].to(logits.device))
        y_pred.extend(pred)
    y_pred = extract_texts(y_pred, texts, offset, id2label)
    return y_pred


def extract_texts(labels, texts, offsets, id2label):
    start = False
    output = []
    for text, offset, label in zip(texts, offsets, labels):
        line = {}
        entity = collections.defaultdict(list)
        seq_len = len(offset)
        label = label[:seq_len]
        for i, v in enumerate(label):
            if v % 2 == 1:
                label = id2label[(v - 1) // 2]
                entity[label].append([i])
                last_start = label
                start = True
            elif start and v != 0:
                label = id2label[(v - 1) // 2]
                if label == last_start:
                    entity[label][-1].append(i)
            else:
                start = False

        for k, v in entity.items():
            if len(v) > 0:
                for j in v:
                    start = j[0]
                    end = j[-1]
                    mapping_start = offset[start][0]
                    mapping_end = offset[end][1]
                    entity_text = text[mapping_start:mapping_end]
                    if entity_text not in line:
                        line[entity_text] = {"entity": entity_text, 'label': k,
                                             'spans': [(mapping_start, mapping_end - 1)]}
                    else:
                        line[entity_text]['spans'].append((mapping_start, mapping_end - 1))
        output.append(list(line.values()))

    return output


def extract_biaffine(labels, texts, offsets, id2label):
    import torch
    output = []
    for text, offset, label in zip(texts, offsets, labels):
        line = {}
        seq_len = len(offset)
        indices_mat = torch.triu(torch.ones((seq_len, seq_len)), diagonal=0, out=None).to(labels[0].device)
        indices = torch.where(indices_mat > 0)
        logits = label[indices[0], indices[1]].argmax(-1)

        label_mapping = {}
        for i in range(seq_len):
            for j in range(i, seq_len):
                label_mapping[len(label_mapping)] = (i, j)

        for idx, value in enumerate(logits):
            if value > 0:
                start, end = label_mapping[idx]
                mapping_start = offset[start][0]
                mapping_end = offset[end][1]
                entity_text = text[mapping_start:mapping_end]
                if entity_text in line:
                    line[entity_text]['spans'].append((mapping_start, mapping_end - 1))
                else:
                    line[entity_text] = {"entity": entity_text, 'label': id2label[value - 1],
                                         'spans': [(mapping_start, mapping_end - 1)]}
        output.append(list(line.values()))

    return output


def extract_span(start_logits, end_logits, texts, offset, id2label):
    ret = []
    for s, e, t, o in zip(start_logits, end_logits, texts, offset):
        print(t)
        s = s.cpu().numpy()[1:-1]
        e = e.cpu().numpy()[1:-1]
        o = o[1:-1]
        for i, s_l in enumerate(s):
            if s_l == 0:
                continue
            for j, e_l in enumerate(e[i:]):
                # if t != '装修新，细节好，带空气净化，灯电视窗帘可智能语音控制，干净卫生，毛巾有密封包装，拖鞋除了房间用的还有淋浴专用的，性价比高很好。':
                #     continue
                if s_l == e_l:
                    mapping_start = o[i][0]
                    mapping_end = o[i + j][1]
                    entity_text = t[mapping_start:mapping_end]
                    ret.append((t, entity_text, id2label[s_l - 1], i, i + j))
                    print((t, entity_text, id2label[s_l - 1], i, i + j))
                    break
    return ret
