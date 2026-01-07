seq_len=512
model=GPT4TS

for percent in 100
do
for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --patch_len 16 \
    --stride 8 \
    --all 1 \
    --llm_layers 6 \
    --itr 1 \
    --model $model \
    --patience 3 \
    --tmax 10 \
    --is_gpt 1

done
done