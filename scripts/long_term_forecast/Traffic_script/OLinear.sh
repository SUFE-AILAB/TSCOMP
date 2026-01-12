
model_name=OLinear

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)

d_models=(512 512 512 512)

cuda_ids1=(0 0 0 0)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --q_mat_dir traffic_${seq_len}_ratio0.7.npy \
      --q_out_mat_dir traffic_${pred_len}_ratio0.7.npy \
      --model_id Traffic_OLinear_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 32 \
      --learning_rate 5e-4 \
      --itr 1 \
      --e_layers 3 \
      --train_epochs 50 \
      --patience 5 \
      --lradj cosine \
      --loss WeightedL1 \
      --dropout 0.0 

done