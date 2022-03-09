
from utils.hyperopt_run import  *
from generate_job_parameters import *
import os

if __name__ == '__main__':

    fold = 'first_run_class'
    jobs = os.listdir(fold)
    jobs.sort()
    print(jobs)
    for i in [-1]:
        job = jobs[i]
        job_params = load_obj(job,folder=f'{fold}/')
        h_space = job_params['h_space']
        VI_params = job_params['VI_params']
        training_params = job_params['training_params']
        if training_params['regression']:
            e=experiment_regression_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
        else:
            e=experiment_classification_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
        e.run()