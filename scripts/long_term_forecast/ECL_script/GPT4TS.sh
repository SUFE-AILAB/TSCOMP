export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS

for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 2048 \
    --learning_rate 0.0001 \
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
    --itr 1 \
    --model $model \
    --tmax 10 \
    --is_gpt 1
done
done