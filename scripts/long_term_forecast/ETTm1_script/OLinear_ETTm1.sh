
model_name=OLinear

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)

d_models=(512 512 512 512)

cuda_ids1=(0 0 0 0)

learning_rate=(1e-4 1e-4 1e-4 1e-4)
dropout=(0.1 0.1 0.1 0.2)


for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --q_mat_dir ETTm1_${seq_len}_ratio0.6.npy\
      --q_out_mat_dir ETTm1_${pred_len}_ratio0.6.npy\
      --model_id ETTm1_OLinear_${seq_len}_${pred_len}_${learning_rate[i]} \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model ${d_models[i]} \
      --d_ff ${d_models[i]} \
      --batch_size 32 \
      --learning_rate ${learning_rate[i]} \
      --itr 1 \
      --e_layers 2 \
      --train_epochs 30 \
      --patience 8 \
      --lradj type1 \
      --loss WeightedL1 \
      --dropout ${dropout[i]} 
done