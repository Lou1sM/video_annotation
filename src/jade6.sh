#!/bin/bash

#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --time=144:00:00
#SBATCH -J WikiLong
#SBATCH --gres=gpu:1

layers=3
rnn='gru'
teacher_forcing_ratio=0.7
i=0
name_prefix='jade6'
for learning_rate in 0.001 0.0003 0.0001 ; 
do
    for rnn_size in 1000 1500 2000 ; 
    do
        for lyrs in 2 3 ; 
        do
            for weight_decay in 0.1 0.01 ; 
            do
                for lmbda_norm in 0.5 1 1.5 ; 
                do
                    CUDA_VISIBLE_DEVICES=1 python3 main.py --enc_layers ${layers} --dec_layers ${layers} --enc_rnn ${rnn} --dec_rnn ${rnn} --teacher_forcing_ratio ${teacher_forcing_ratio} --learning_rate ${learning_rate} --enc_size $(expr 1.5*${rnn_size} | bc) --dec_size $(expr 1.25*${rnn_size} | bc ) --enc_layers ${lyrs} --dec_layers ${lyrs} --lmbda_norm $(expr 1.0*${lmbda_norm} | bc ) --i3d --exp_name  ${name_prefix}-${i}
                    i=$(expr ${i}+1 | bc)
                done
            done
        done
    done
done


