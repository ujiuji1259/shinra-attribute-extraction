import argparse
import sys
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from dataset import ShinraData, NerDataset, ner_collate_fn
from model import BertForMultilabelNER, create_pooler_matrix

import os

device = "cuda:1" if torch.cuda.is_available() else "cpu"


def ner_for_shinradata(model, tokenizer, shinra_dataset, device):
    processed_data = shinra_dataset.ner_inputs
    dataset = NerDataset(processed_data, tokenizer)
    total_preds, _ = predict(model, dataset, device, sent_wise=True)

    shinra_dataset.add_nes_from_iob(total_preds)

    return shinra_dataset


def predict(model, dataset, device, sent_wise=False):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=ner_collate_fn)

    total_preds = []
    total_trues = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            input_ids = inputs["tokens"]
            word_idxs = inputs["word_idxs"]

            labels = inputs["labels"]

            input_ids = pad_sequence([torch.tensor(t) for t in input_ids], padding_value=0, batch_first=True).to(device)
            attention_mask = input_ids > 0
            pooling_matrix = create_pooler_matrix(input_ids, word_idxs, pool_type="head").to(device)

            preds = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_idxs=word_idxs,
                pooling_matrix=pooling_matrix
            )

            total_preds.append(preds)
            # test dataの場合truesは使わないので適当にpredsを入れる
            total_trues.append(labels if labels[0] is not None else preds)

    attr_num = len(total_preds[0])
    total_preds = [[pred for preds in total_preds for pred in preds[attr]] for attr in range(attr_num)]
    total_trues = [[true for trues in total_trues for true in trues[attr]] for attr in range(attr_num)]

    if sent_wise:
        total_preds = [[total_preds[attr][idx] for attr in range(attr_num)] for idx in range(len(total_preds[0]))]
        total_trues = [[total_trues[attr][idx] for attr in range(attr_num)] for idx in range(len(total_trues[0]))]

    return total_preds, total_trues


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, help="Specify input path in SHINRA2020")
    parser.add_argument("--model_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--output_path", type=str, help="Specify attribute_list path in SHINRA2020")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arg()

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    # dataset = [ShinraData(), ....]
    dataset = ShinraData.from_shinra2020_format(Path(args.input_path))
    dataset = [d for idx, d in enumerate(dataset) if idx < 20 and d.nes is not None]

    model = BertForMultilabelNER(bert, len(dataset[0].attributes))
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    dataset = [ner_for_shinradata(model, tokenizer, d, device) for d in dataset]
    with open(args.output_path, "w") as f:
        f.write("\n".join([json.dumps(ne, ensure_ascii=False) for d in dataset for ne in d.nes]))
