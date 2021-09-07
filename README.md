# shinra-attribute-extraction

## データ
SHINRA2020での[前処理済みデータ](http://shinra-project.info/shinra2020jp/data_download/)の森羅2020-JPタスクの学習・ターゲットデータ（トークナイズ済み, Mecab(IPA辞書)&BPE使用, 東北大BERT対応)を使用しています。

## モデル
事前学習済みモデルとして[東北大BERT](https://github.com/cl-tohoku/bert-japanese)を使用しています．
BERTの上に属性ごとに独立した分類層を乗せています．

## 環境

- pytorch
- transformers>=3.0.1
- fugashi
- seqeval
- mlflow

### Docker
[こちら](https://github.com/frisk0zisan/docker_pytorch)をご利用いただけます．



## 学習
`sh train.sh`

※ `model_path`はディレクトリです．validation setで最大精度のモデルと最終エポックのモデルを保存します．

### train.shの例
```bash
python train.py \
    --input_path /path/to/Target_Category \
    --model_path /path/to/model_directory \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 50 \
    --grad_acc 1 \
    --grad_clip 1.0 
```

## 予測
`sh predict.sh`.   
前処理済みのデータ（１カテゴリ）を入力に，森羅2020の出力形式で予測結果を出力.   
※ `model_path`はモデルファイルへのパスです．

### predict.shの例
```bash
python predict.py \
    --input_path /path/to/Target_Category \
    --model_path /path/to/model_file \
    --output_path /path/to/output_file
```
