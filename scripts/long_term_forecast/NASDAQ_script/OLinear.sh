
model_name=OLinear

pred_lens=(24 36 48 60)
seq_lens=(36 36 36 36)


d_models=(512 512 512 512)

cuda_ids1=(1 1 1 1)

epochs=(50 50 50 50)
lradj=(type1 type1 type1 type1)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/nasdaq/ \
      --data_path nasdaq.csv \
      --q_mat_dir nasdaq_${seq_len}_ratio0.7.npy \
      --q_out_mat_dir nasdaq_${pred_len}_ratio0.7.npy \
      --model_id Nasdaq_OLinear_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 12 \
      --dec_in 12 \
      --c_out 12 \
      --des 'Exp' \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 4 \
      --learning_rate 1e-4 \
      --itr 1 \
      --e_layers 3 \
      --train_epochs ${epochs[i]} \
      --patience 5 \
      --lradj ${lradj[i]} \
      --loss WeightedL1 \
      --dropout 0.0 

done