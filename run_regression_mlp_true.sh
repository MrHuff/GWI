CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 0-3 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=0 > myoutput_1.txt &
CUDA_VISIBLE_DEVICES=1 taskset --cpu-list 4-7 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=1 > myoutput_2.txt &
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 8-11 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=2 > myoutput_3.txt &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list 12-15 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=3 > myoutput_4.txt &
CUDA_VISIBLE_DEVICES=4 taskset --cpu-list 16-19 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=4 > myoutput_5.txt &
CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 20-23 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=5 > myoutput_6.txt &
CUDA_VISIBLE_DEVICES=6 taskset --cpu-list 24-27 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=6 > myoutput_7.txt &
CUDA_VISIBLE_DEVICES=7 taskset --cpu-list 28-31 python serious_run_jade.py --job_folder="50_1000_reg_mlp_True_Tanh" --total_chunks=8 --parallel_jobs=1 --chunk_idx=7 > myoutput_8.txt &
