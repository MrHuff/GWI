import torch

from utils.hyperopt_run import *
from simulate_data.unit_test_data import *
import seaborn as sns
import matplotlib.pyplot as plt
from GP_baseline.gp_baseline_exact import *
from GP_baseline.gp_baseline_vi import *
sns.set()
#TODO: upgrade m_Q wtf haha
nn_params = {
    'layers_x': [8,8],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': torch.tanh,
    'output_dim': 1,
}
VI_params={
    'q_kernel':'r_param_scaling',
    'p_kernel':'rbf',
    'm_p':0.0,
    'reg':1e-2,
    'APQ':True,
    'parametrize_Z': True

}


training_params = {
                    'model_name':'GWI',
                   'patience': 10,
                   'device': 'cuda:0',
                   'epochs':500,
                    'fold':0,
                    'seed':1,
                    'savedir':'CIFAR_TEST',
                    'hyperits':1,
                    'val_factor':0.05,
                    'output_classes':10,
                    'image_size':32,
                    'cdim':3,
                    'dataset':'CIFAR10',
                    'm_q_choice':'CNN' #CNN,kernel_sum
                   }

h_space={
    'depth_x':[3],
    'width_x':[8,16,32],
    'bs':[500],
    'lr':[1e-2],
    'm_P':[0.0],
    'sigma':[1e-4],
    'transformation':[torch.relu],
    'depth_fc':[1],
    'm':[250]
}
if __name__ == '__main__':

    torch.random.manual_seed(np.random.randint(0,100000))
    e=experiment_classification_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
    e.run()

