# shinra-attribute-extraction

## データ
SHINRA2020の[東北大BERT](https://github.com/cl-tohoku/bert-japanese)での[前処理済みデータ](http://shinra-project.info/shinra2020jp/data_download/)

## 学習
`sh train.sh`

※ `model_path`はディレクトリです．validation setで最大精度のモデルと最終エポックのモデルを保存します．

### train.shの例
```bash
python train.py \
    --input_path /path/to/Event/Event_Other \
    --model_path /path/to/model_dir \
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
    --input_path /path/to/Event/Event_Other \
    --model_path /path/to/model_file \
    --output_path output.jsonl
```
