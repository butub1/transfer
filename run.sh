srun --mpi=pmi2 -p VI_AIC_1080TI -n 8 --gres=gpu:8 --ntasks-per-node=2 python3 -u $1
