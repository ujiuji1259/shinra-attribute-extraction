import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np


def create_pooler_matrix(input_ids, word_idxs, pool_type="head"):
    bsz, subword_len = input_ids.size()
    max_word_len = max([len(w) for w in word_idxs])
    pooler_matrix = torch.zeros(bsz * max_word_len * subword_len)

    if pool_type == "head":
        pooler_idxs = [subword_len * max_word_len * batch_offset +  subword_len * word_offset + w
            for batch_offset, word_idx in enumerate(word_idxs) for word_offset, w in enumerate(word_idx[:-1])]
        pooler_matrix.scatter_(0, torch.LongTensor(pooler_idxs), 1)
        return pooler_matrix.view(bsz, max_word_len, subword_len)

    elif pool_type == "average":
        pooler_idxs = [subword_len * max_word_len * batch_offset +  subword_len * word_offset + w / (word_idx[word_offset+1] - word_idx[word_offset])
            for batch_offset, word_idx in enumerate(word_idxs)
            for word_offset, _ in enumerate(word_idx[:-1])
            for w in range(word_idx[word_offset], word_idx[word_offset+1])]
        pooler_matrix.scatter_(0, torch.LongTensor(pooler_idxs), 1)
        return pooler_matrix.view(bsz, max_word_len, subword_len)


class BertForMultilabelNER(nn.Module):
    def __init__(self, bert, attribute_num, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        output_layer = [nn.Linear(768, 768) for i in range(attribute_num)]
        self.output_layer = nn.ModuleList(output_layer)

        self.relu = nn.ReLU()

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        classifiers = [nn.Linear(768, 3) for i in range(attribute_num)]
        self.classifiers = nn.ModuleList(classifiers)


    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        word_idxs=None,
        pooling_matrix=None,
    ):
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_matrix=pooling_matrix)[0]
        #labels = [torch.argmax(logit.detach().cpu(), dim=-1) for logit in logits]
        labels = [self.viterbi(logit.detach().cpu()) for logit in logits]

        truncated_labels = [[label[:len(word_idx)-1] for label, word_idx in zip(attr_labels, word_idxs)] for attr_labels in labels]

        return truncated_labels


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pooling_matrix=None,
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # create word-level representations using pooler matrix
        sequence_output = torch.bmm(pooling_matrix, sequence_output)
        sequence_output = self.dropout(sequence_output)

        # hiddens = [self.relu(layer(sequence_output)) for layer in self.output_layer]
        # logits = [classifier(hiddens) for classifier, hiddens in zip(self.classifiers, hiddens)]
        logits = [classifier(sequence_output) for classifier in self.classifiers]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = 0

            for label, logit in zip(labels, logits):
                loss += loss_fct(logit.view(-1, 3), label.view(-1)) / len(labels)

        output = (logits, ) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    def viterbi(self, logits, penalty=float('inf')):
        num_tags = 3

        # 0: O, 1: B, 2: I
        penalties = torch.zeros((num_tags, num_tags))
        penalties[0][2] = penalty

        all_preds = []
        for logit in logits:
            pred_tags = [0]
            for l in logit:
                transit_penalty = penalties[pred_tags[-1]]
                l = l - transit_penalty
                tag = torch.argmax(l, dim=-1)
                pred_tags.append(tag.item())
            all_preds.append(pred_tags[1:])
        return all_preds
