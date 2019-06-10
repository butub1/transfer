srun --mpi=pmi2 -p  VI_AIC_1080TI -w SH-IDC1-10-5-34-131 -n 1 --gres=gpu:8 --ntasks-per-node=1 python3 -u $1
