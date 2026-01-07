

seq_len=96
model=GPT4TS

for pred_len in 96 192 336 720
do
for lr in 0.0001
do

python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 96 \
    --pred_len $pred_len \
    --lradj type3 \
    --learning_rate $lr \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --patch_len 16 \
    --stride 8 \
    --llm_layers 6 \
    --model $model \
    --is_gpt 1

done
done