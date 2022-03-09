import argparse
import os

from utils.hyperopt_run import *
from generate_job_parameters import *
parser = argparse.ArgumentParser()

parser.add_argument('--idx', type=int, default=0, help='cdim')
parser.add_argument('--job_folder', type=str, default='', help='cdim')


if __name__ == '__main__':
    input  = vars(parser.parse_args())
    idx = input['idx']
    fold = input['job_folder']
    jobs = os.listdir(fold)
    jobs.sort()
    print(jobs[idx])
    experiment_params = load_obj(jobs[idx],folder=f'{fold}/')
    c = experiment_object(**experiment_params)
    c.debug_mode=True
    c.run_experiments()