export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS

for percent in 100
do
for pred_len in 96 192 336 720
do

python main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
    --data ett_m \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --patch_len 16 \
    --stride 16 \
    --percent $percent \
    --llm_layers 6 \
    --itr 1 \
    --model $model \
    --is_gpt 1
done
done