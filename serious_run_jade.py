import argparse
import os
import numpy as np
import torch.cuda

from utils.hyperopt_run import *
from generate_job_parameters import *
from torch.multiprocessing import Pool
# torch.multiprocessing.set_start_method("spawn")
parser = argparse.ArgumentParser()

parser.add_argument('--chunk_idx', type=int, default=0, help='cdim')
parser.add_argument('--total_chunks', type=int, default=8, help='cdim')
parser.add_argument('--parallel_jobs', type=int, default=2, help='cdim')
parser.add_argument('--job_folder', type=str, default='', help='cdim')

def run_func(job_params):
    h_space = job_params['h_space']
    VI_params = job_params['VI_params']
    training_params = job_params['training_params']
    if training_params['regression']:
        e = experiment_regression_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
    else:
        e = experiment_classification_object(hyper_param_space=h_space, VI_params=VI_params,
                                             train_params=training_params)
    e.run()
    del e
    torch.cuda.empty_cache()

def check_completed_jobs(in_fold,jobs):
    uncomplete_jobs = []
    for j in jobs:
        job_dict = load_obj(j,f'{in_fold}/')
        train_params  = job_dict['training_params']
        max_its = train_params['hyperits']
        model_name = train_params['model_name']
        savedir = train_params['savedir']
        dataset = train_params['dataset']
        fold = train_params['fold']
        seed = train_params['seed']
        save_path = f'{savedir}/{dataset}_seed={seed}_fold_idx={fold}_model={model_name}/'
        path = save_path +  f'best_model_{max_its-1}.pt'
        if not os.path.exists(path):
            uncomplete_jobs.append(j)

    return uncomplete_jobs


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn")

    input  = vars(parser.parse_args())
    chunk_idx = input['chunk_idx']
    fold = input['job_folder']
    total_chunks = input['total_chunks']
    parallel_jobs = input['parallel_jobs']

    jobs = os.listdir(fold)
    jobs = check_completed_jobs(fold,jobs)

    jobs = sorted(jobs, key=lambda x: int(x[-8]))
    job_chunks_list = np.array_split(jobs, total_chunks)

    job_chunk=job_chunks_list[chunk_idx]
    chunked_input=[]
    for el in job_chunk:
        loaded = load_obj(el, folder=f'{fold}/')
        chunked_input.append(loaded)
    if parallel_jobs==1:
        for el in chunked_input:
            print(el)
            run_func(el)
    else:
        with Pool(processes = parallel_jobs) as p:   # Paralleizing over 2 GPUs
            results = p.map(run_func,chunked_input)