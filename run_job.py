
from utils.hyperopt_run import  *
from generate_job_parameters import *
import os

if __name__ == '__main__':

    fold = 'all_gpu_real'
    jobs = os.listdir(fold)
    jobs.sort()
    for job in jobs:
        experiment_params = load_obj(job,folder=f'{fold}/')
        c = experiment_object(**experiment_params)
        c.debug_mode=True
        c.run_experiments()