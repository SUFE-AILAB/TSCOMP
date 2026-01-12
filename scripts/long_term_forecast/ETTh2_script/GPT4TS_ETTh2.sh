2export CUDA_VISIBLE_DEVICES=0

seq_len=336
model=GPT4TS

for pred_len in 96 192 336 720
do

python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$model'_'$seq_len'_'$pred_len \
    --data ETTh2 \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 1 \
    --enc_in 7 \
    --c_out 7 \
    --patch_len 16 \
    --stride 8 \
    --llm_layers 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --pretrain 1 \
    --is_gpt 1

done