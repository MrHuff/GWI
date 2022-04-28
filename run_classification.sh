CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 0-3 python serious_run.py --idx=0 --job_folder="1000_epoch_class_False"  > myoutput_0.txt &
CUDA_VISIBLE_DEVICES=3 taskset --cpu-list 4-7 python serious_run.py --idx=1 --job_folder="1000_epoch_class_False"  > myoutput_1.txt &
CUDA_VISIBLE_DEVICES=4 taskset --cpu-list 8-11 python serious_run.py --idx=2 --job_folder="1000_epoch_class_False"  > myoutput_2.txt &
CUDA_VISIBLE_DEVICES=5 taskset --cpu-list 12-15 python serious_run.py --idx=3 --job_folder="1000_epoch_class_False"  > myoutput_3.txt &
CUDA_VISIBLE_DEVICES=6 taskset --cpu-list 16-19 python serious_run.py --idx=4 --job_folder="1000_epoch_class_False"  > myoutput_4.txt &
CUDA_VISIBLE_DEVICES=7 taskset --cpu-list 20-23 python serious_run.py --idx=5 --job_folder="1000_epoch_class_False"  > myoutput_5.txt &