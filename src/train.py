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
from seqeval.metrics import f1_score, classification_report
import mlflow
from sklearn.model_selection import train_test_split

from dataset import ShinraData
from dataset import NerDataset, ner_collate_fn, decode_iob
from model import BertForMultilabelNER, create_pooler_matrix
from predict import predict

device = "cuda:1" if torch.cuda.is_available() else "cpu"


class EarlyStopping():
   def __init__(self, patience=0, verbose=0):
       self._step = 0
       self._score = - float('inf')
       self.patience = patience
       self.verbose = verbose

   def validate(self, score):
       if self._score > score:
           self._step += 1
           if self._step > self.patience:
               if self.verbose:
                   print('early stopping')
               return True
       else:
           self._step = 0
           self._score = score

       return False


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, help="Specify input path in SHINRA2020")
    parser.add_argument("--model_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--lr", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--bsz", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--epoch", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--grad_acc", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--grad_clip", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--note", type=str, help="Specify attribute_list path in SHINRA2020")

    args = parser.parse_args()

    return args

def evaluate(model, dataset, attributes, args):
    total_preds, total_trues = predict(model, dataset, device)
    total_preds = decode_iob(total_preds, attributes)
    total_trues = decode_iob(total_trues, attributes)

    f1 = f1_score(total_trues, total_preds)
    return f1


def train(model, train_dataset, valid_dataset, attributes, args):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = get_scheduler(
    #     args.bsz, args.grad_acc, args.epoch, args.warmup, optimizer, len(train_dataset))

    early_stopping = EarlyStopping(patience=10, verbose=1)

    losses = []
    for e in range(args.epoch):
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=ner_collate_fn, shuffle=True)
        bar = tqdm(total=len(train_dataset))

        total_loss = 0
        model.train()
        for step, inputs in enumerate(train_dataloader):
            input_ids = inputs["tokens"]
            word_idxs = inputs["word_idxs"]
            labels = inputs["labels"]

            labels = [pad_sequence([torch.tensor(l) for l in label], padding_value=-1, batch_first=True).to(device)
                for label in labels]

            input_ids = pad_sequence([torch.tensor(t) for t in input_ids], padding_value=0, batch_first=True).to(device)
            attention_mask = input_ids > 0
            pooling_matrix = create_pooler_matrix(input_ids, word_idxs, pool_type="head").to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pooling_matrix=pooling_matrix)

            loss = outputs[0]
            loss.backward()

            total_loss += loss.item()
            mlflow.log_metric("Trian batch loss", loss.item(), step=(e+1) * step)

            bar.set_description(f"[Epoch] {e + 1}")
            bar.set_postfix({"loss": loss.item()})
            bar.update(args.bsz)

            if (step + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

        losses.append(total_loss / (step+1))
        mlflow.log_metric("Trian loss", losses[-1], step=e)

        valid_f1 = evaluate(model, valid_dataset, attributes, args)
        mlflow.log_metric("Valid F1", valid_f1, step=e)

        if early_stopping._score < valid_f1:
            torch.save(model.state_dict(), args.model_path + "best.model")


        if e + 1 > 30 and early_stopping.validate(valid_f1):
            break


if __name__ == "__main__":
    args = parse_arg()

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    # dataset = [ShinraData(), ....]
    dataset = ShinraData.from_shinra2020_format(Path(args.input_path))
    dataset = [d for d in dataset if d.nes is not None]

    model = BertForMultilabelNER(bert, len(dataset[0].attributes)).to(device)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1)
    train_dataset = NerDataset([d for train_d in train_dataset for d in train_d.ner_inputs], tokenizer)
    valid_dataset = NerDataset([d for valid_d in valid_dataset for d in valid_d.ner_inputs], tokenizer)

    mlflow.start_run()
    mlflow.log_params(vars(args))
    train(model, train_dataset, valid_dataset, dataset[0].attributes, args)
    torch.save(model.state_dict(), args.model_path + "last.model")
    mlflow.end_run()
