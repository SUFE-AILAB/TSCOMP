
model_name=OLinear

pred_lens=(24 36 48 60)
seq_lens=(36 36 36 36)

d_models=(256 256 256 256)

cuda_ids1=(1 1 1 1)
epochs=(50 50 50 30)
lradj=(type3 type3 type3 type3)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}


    python -u run.py \
          --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ILI/ \
      --data_path national_illness.csv \
      --q_mat_dir ILI_${seq_len}_ratio0.70.npy \
      --q_out_mat_dir ILI_${pred_len}_ratio0.70.npy \
      --model_id ILI_OLinear_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 4 \
      --learning_rate 1e-4 \
      --itr 1 \
      --e_layers 2 \
      --train_epochs ${epochs[i]} \
      --patience 10 \
      --lradj ${lradj[i]} \
      --loss WeightedL1 \
      --dropout 0.0 

done