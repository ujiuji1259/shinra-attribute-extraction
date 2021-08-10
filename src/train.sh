python train.py \
    --input_path /data1/ujiie/shinra/tohoku_bert/Event/Event_Other \
    --model_path /home/is/ujiie/shinra-attribute-extraction/models/ \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 50 \
    --grad_acc 1 \
    --grad_clip 1.0 \
