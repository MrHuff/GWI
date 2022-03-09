import pickle
import torch
import os
def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

EPOCHS=500
def generate_classification_jobs(job_name):

    if not os.path.exists(job_name):
        os.makedirs(job_name)

    dataset = ['FashionMNIST','CIFAR10']
    VI_params = {
        'q_kernel': 'r_param',
        'p_kernel': 'rbf',
        'sigma': 1.0,
        'reg': 1e-2,
        'y_var': 10.0,
        'APQ': False
    }
    h_space = {
        'depth_x': [3],
        'width_x': [8, 16, 32,64],
        'bs': [250,500, 1000],
        'lr': [1e-2, 1e-3],
        'm_P': [0.0, 0.5, 1.0],
        'sigma': [1e-3, 1e-2, 1e-4],
        'transformation': [torch.tanh, torch.relu],
        'm': [10, 50, 100],
        'depth_fc':[1,2,3]

    }
    training_params = {
        'patience': 10,
        'device': 'cuda:0',
        'epochs': EPOCHS,
        'model_name': 'GWI',
        'savedir': f'{job_name}_results',
        'seed': 0,
        'fold': 0,
        'hyperits': 20,
        'val_factor':0.05,
        'output_classes':10,
        'image_size':32,
        'cdim':3,
        'regression': False
    }
    for ds,c in zip(dataset,[1,3]):
        training_params['dataset'] = ds
        training_params['cdim'] = c
        for i in range(5):
            training_params['fold'] = i
            job_dict = {'training_params': training_params, 'h_space': h_space, 'VI_params': VI_params}
            with open(f'{job_name}/dataset={ds}_fold={i}.pickle', 'wb') as handle:
                pickle.dump(job_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_regression_jobs(job_name):
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    dataset = ['housing', 'concrete', 'energy', 'power', 'wine', 'yacht', 'naval', 'KIN8NM']
    VI_params = {
        'q_kernel': 'r_param',
        'p_kernel': 'rbf',
        'sigma': 1.0,
        'reg': 1e-2,
        'y_var': 10.0,
        'APQ': False
    }
    h_space = {
        'depth_x': [2],
        'width_x': [10],
        'bs': [100,250,500,1000],
        'lr': [1e-2,1e-3],
        'm_P': [0.0, 0.5,1.0],
        'sigma': [1e-3,1e-2,1e-4],
        'transformation': [torch.tanh,torch.relu],
        'm':[10,50,100],
    }
    training_params = {
        'patience': 50,
        'device': 'cuda:0',
        'epochs': EPOCHS,
        'model_name': 'GWI',
        'savedir': f'{job_name}_results',
        'seed': 0,
        'fold':0,
        'hyperits': 20,
        'regression':True
    }
    for ds  in dataset:
        training_params['dataset'] = ds
        for i in range(10):
            training_params['fold'] = i
            job_dict={'training_params':training_params,'h_space':h_space,'VI_params':VI_params}
            with open(f'{job_name}/dataset={ds}_fold={i}.pickle', 'wb') as handle:
                pickle.dump(job_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    generate_classification_jobs('first_run_class')
    generate_regression_jobs('first_run_reg')