import torch

from utils.hyperopt_run import *
from simulate_data.unit_test_data import *
import seaborn as sns
import matplotlib.pyplot as plt
from GP_baseline.gp_baseline_exact import *
from GP_baseline.gp_baseline_vi import *
import shutil

sns.set()
# TODO: upgrade m_Q wtf haha
nn_params = {
    'layers_x': [8, 8],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': torch.tanh,
    'output_dim': 1,
}
VI_params = {
    'q_kernel': 'r_param_scaling',
    'p_kernel': 'rbf',
    'm_p': 0.0,
    'reg': 1e-2,
    'APQ': True,

}

training_params = {
    'model_name': 'GWI',
    'patience': 1,
    'device': 'cuda:0',
    'epochs': 1,
    'fold': 0,
    'seed': 1,
    'savedir': 'CIFAR_TEST',
    'hyperits': 1,
    'val_factor': 0.05,
    'output_classes': 10,
    'image_size': 32,
    'cdim': 3,
    'dataset': 'CIFAR10',
    'm_q_choice': 'CNN',  # CNN,kernel_sum
    'init_its': 10
}

h_space = {
    'depth_x': [3],
    'width_x': [64],
    'bs': [1000],
    'lr': [1e-2],
    'm_P': [0.1],
    'sigma': [1e-4],
    'transformation': [torch.relu],
    'depth_fc': [1],
    'm_factor': [1.0],
    'parametrize_Z': [True],
    'use_all_m': [False],
    'x_s':[50]

}
if __name__ == '__main__':
    if os.path.exists('CIFAR_TEST'):
        shutil.rmtree('CIFAR_TEST')
    torch.random.manual_seed(np.random.randint(0, 100000))
    e = experiment_classification_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
    e.run()
