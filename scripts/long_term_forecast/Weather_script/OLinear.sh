
model_name=OLinear

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)

d_models=(512 512 512 512)

cuda_ids1=(0 0 0 0)
dropout=(0.0 0.0 0.0 0.0)
learning_rate=(1e-3 1e-3 1e-3 1e-3)

layer_num=2

lradj=(type3 type3 type3 type3)

for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --q_mat_dir weather_${seq_len}_ratio0.7.npy \
      --q_out_mat_dir weather_${pred_len}_ratio0.7.npy \
      --model_id Weather_OLinear_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --label_len 0 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 32 \
      --learning_rate ${learning_rate[i]} \
      --itr 1 \
      --e_layers $layer_num \
      --train_epochs 30 \
      --patience 5 \
      --lradj ${lradj[i]} \
      --loss WeightedL1 \
      --dropout ${dropout[i]} 

done