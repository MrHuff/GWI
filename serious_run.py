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
    job_params = load_obj(jobs[idx], folder=f'{fold}/')
    h_space = job_params['h_space']
    VI_params = job_params['VI_params']
    training_params = job_params['training_params']
    if training_params['regression']:
        e = experiment_regression_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
    else:
        e = experiment_classification_object(hyper_param_space=h_space, VI_params=VI_params,
                                             train_params=training_params)
    e.run()

