import pickle
import torch
import os
def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)
reg_EPOCHS=7500
class_EPOCHS=500
PARAM=True

def generate_classification_jobs(job_name):
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    dataset = ['FashionMNIST','CIFAR10']
    VI_params = {
        'q_kernel': 'r_param_scaling',
        'p_kernel': 'rbf',
        'sigma': 1.0,
        'reg': 1e-2,
        'y_var': 10.0,
        'APQ': True,
    'parametrize_Z':PARAM

    }
    h_space = {
        'depth_x': [3],
        'width_x': [8, 16, 32,64],
        'bs': [250,500, 1000],
        'lr': [1e-2,1e-1],
        'm_P': [0.1],
        'sigma': [1e-3, 1e-2, 1e-4,1e-5,1e-6],
        'transformation': [torch.tanh, torch.relu],
        'depth_fc':[1,2,3],
        'm_factor': [1.0]

    }
    training_params = {
        'patience': 50,
        'device': 'cuda:0',
        'epochs': class_EPOCHS,
        'model_name': 'GWI',
        'savedir': f'{job_name}_results',
        'seed': 0,
        'fold': 0,
        'hyperits': 20,
        'val_factor':0.05,
        'output_classes':10,
        'image_size':32,
        'cdim':3,
        'regression': False,
        'm_q_choice': 'CNN',
        'init_its':100
    }
    for ds,c in zip(dataset,[1,3]):
        training_params['dataset'] = ds
        training_params['cdim'] = c
        for i in range(3):
            training_params['fold'] = i
            job_dict = {'training_params': training_params, 'h_space': h_space, 'VI_params': VI_params}
            with open(f'{job_name}/dataset={ds}_fold={i}.pickle', 'wb') as handle:
                pickle.dump(job_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_regression_jobs(job_name):
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    # dataset = ['boston', 'concrete', 'energy','KIN8NM', 'power','protein' ,'wine', 'yacht', 'naval']
    # use_all_m = [False,False,False,False,False,False,False,True,False]
    use_all_m = [False,False,False,False,True]
    dataset = ['protein','energy','power','naval','yacht']
    init_it_list = [100]*len(dataset)
    VI_params = {
        'q_kernel': 'r_param_simple',
        'p_kernel': 'rbf',
        'sigma': 1.0,
        'reg': 1e-2,
        'y_var': 10.0,
        'APQ': True,
    'parametrize_Z':PARAM

    }
    h_space = {
        'depth_x': [2],
        'width_x': [10],
        'bs': [100,250,500,1000,2500],
        'lr': [1e-2],
        'm_P': [0.0, 0.5,1.0],
        'sigma': [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7],
        'transformation': [torch.tanh,torch.relu],
        'm_factor':[0.5,0.75,1.0,1.5,2.0]

    }
    training_params = {
        'patience': 50,
        'device': 'cuda:0',
        'epochs': reg_EPOCHS,
        'model_name': 'GWI',
        'savedir': f'{job_name}_results',
        'seed': 0,
        'fold':0,
        'hyperits': 30,
        'regression':True,
        'm_q_choice': 'mlp',
        'init_its': 100

    }
    for ds,its,use_all  in zip(dataset,init_it_list,use_all_m):
        training_params['dataset'] = ds
        training_params['init_its'] = its
        training_params['use_all_m']=use_all
        h_space_tmp=h_space
        if use_all:
            h_space_tmp['m_factor']=[1.0]
        for i in range(10):
            training_params['fold'] = i
            job_dict={'training_params':training_params,'h_space':h_space_tmp,'VI_params':VI_params}
            with open(f'{job_name}/dataset={ds}_fold={i}.pickle', 'wb') as handle:
                pickle.dump(job_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # generate_classification_jobs('learned_z_class_5')
    generate_regression_jobs('learned_z_reg_7')