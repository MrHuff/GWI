import copy
import pickle
import torch
import os
import random
def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)
reg_EPOCHS=1000
class_EPOCHS=100
class_BS = 1000
reg_BS = 1000
x_S_reg = 100
x_S_class = 100

def generate_classification_jobs():
    for param in [False]:
        job_name = f'{class_BS}_epoch_class_{param}'
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

        }
        h_space = {
            'depth_x': [3],
            'width_x': [256,512],
            'bs': [class_BS],
            'lr': [1e-2],
            'm_P': [0.1],
            'sigma': [1e-3],
            'transformation': [torch.nn.Tanh(), torch.nn.ReLU(),torch.nn.SELU()],
            'depth_fc':[1],
            'm_factor': [0.5,0.75,1.0],
        'parametrize_Z': [param],
        'use_all_m': [False],
        'x_s':[x_S_class]
        }
        training_params = {
            'patience': 100,
            'device': 'cuda:0',
            'epochs': class_EPOCHS,
            'model_name': 'GWI',
            'savedir': f'{job_name}_results',
            'seed': 0,
            'fold': 0,
            'hyperits': 10,
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

def generate_regression_jobs():
    for m_q in ['mlp']:
        act_list = [torch.nn.Tanh()]
        for act in act_list:
            # for pmz in [True,False]:
            for pmz in [False]:
                job_name = f'new_{x_S_reg}_{reg_BS}_reg_{m_q}_{pmz}_{act._get_name()}'
                if not os.path.exists(job_name):
                    os.makedirs(job_name)
                dataset = ['boston', 'concrete', 'energy','KIN8NM', 'power','protein' ,'wine', 'yacht', 'naval']
                # dataset = ['energy', 'power','protein']
                use_all_m = [True,True,True,False,False,False,True,True,False]
                # use_all_m = [True,False,False]
                # use_all_m = [False]
                # dataset = ['yacht']

                init_it_list = [100]*len(dataset)
                VI_params = {
                    'q_kernel': 'r_param_scaling',
                    'p_kernel': 'rbf',
                    'sigma': 1.0,
                    'reg': 1e-2,
                    'y_var': 10.0,
                    'APQ': True,
                }
                h_space = {
                    'depth_x': [2],
                    'width_x': [10],
                    'bs': [reg_BS] ,
                    'lr': [1e-2],
                    'm_P': [0.0,0.25,0.5,0.75,1.0],
                    'sigma': [1e-6],
                    'transformation': [act] ,
                    'm_factor': [0.5,0.25,1.0,1.25,1.5,1.75,2.0,3.0] if m_q =='mlp' else  [1.0,2.0,3.0,4.0,5.0],
                    # 'm_factor':[0.5,1.0,2.0],
                'parametrize_Z':[pmz],
                        'use_all_m':[False],
                    # 'm_q_choice': ['krr','mlp'],
                    'm_q_choice': [m_q],
                    'x_s': [x_S_reg]

                }
                training_params = {
                    'patience': 1000,
                    'device': 'cuda:0',
                    'epochs': reg_EPOCHS,
                    'model_name': 'GWI',
                    'savedir': f'{job_name}_results',
                    'seed': 0,
                    'fold':0,
                    'hyperits': 30,
                    'regression':True,
                    'init_its': 100

                }
                for ds,its,use_all  in zip(dataset,init_it_list,use_all_m):
                    training_params['dataset'] = ds
                    training_params['init_its'] = its
                    h_space_tmp=copy.deepcopy(h_space)
                    if use_all:
                        h_space_tmp['use_all_m'] = [False,True]
                    if ds in ['concrete','energy']:
                        h_space_tmp['bs']=[100]
                    print(ds)
                    print(h_space_tmp)
                    for i in range(10):
                        training_params['fold'] = i
                        job_dict={'training_params':training_params,'h_space':h_space_tmp,'VI_params':VI_params}
                        with open(f'{job_name}/dataset={ds}_fold={i}.pickle', 'wb') as handle:
                            pickle.dump(job_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # generate_classification_jobs()
    generate_regression_jobs()

    #Reruns: Concrete, NAVAL (WITH KRR)