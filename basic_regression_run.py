import os.path

import torch.nn

from utils.hyperopt_run import *
from simulate_data.unit_test_data import *
import seaborn as sns
import matplotlib.pyplot as plt
from GP_baseline.gp_baseline_exact import *
from GP_baseline.gp_baseline_vi import *
from utils.custom_run import gpr_reference
import shutil
sns.set()
#TODO: upgrade m_Q wtf haha
nn_params = {
    'layers_x': [10,10],
    'cat_size_list': [],
    'transformation': torch.tanh,
    'output_dim': 1,
}
VI_params={
    'q_kernel':'nn_kernel',#'r_param_simple',#'r_param_scaling'
    'p_kernel':'rbf',
    'm_p':0.0,
    'reg':1e-2,
    'r':50,
    'APQ': True,
}
h_space={
    'depth_x':[2],
    'width_x':[10],
    'bs':[1000],
    'lr':[1e-2],
    'm_P':[1.0],
    'sigma':[1e-3],
    'transformation':[torch.nn.Tanh()],
    'm_factor':[1.],
    'parametrize_Z': [False],
    'use_all_m': [True],
    'm_q_choice': ['mlp'],
    'x_s':[0],
#You get negative variance WTF?!
}
#KRR issue is likely related to initialization!
training_params = {

                   'patience': 1000,
                   'device': 'cuda:0',
                   'epochs':1000,
                   'lr':1e-2,
                   'model_name':'GWI',
                   'savedir':'regression_test_3',
                   'seed':0,
                   'hyperits':1,
                    'init_its':250
                   }
if __name__ == '__main__':
    #figure out new r that is able to be proportional to MSE
    for f in [True]:
        dirname =f'regression_test_1_{f}'
        training_params['savedir'] = dirname
        h_space['use_all_m'] = [f]
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        # ['boston', 'concrete', 'energy','KIN8NM', 'power','protein' ,'wine', 'wine', 'naval']
        dataset="energy"
        fold=2
        training_params['fold']=fold
        training_params['dataset']=dataset
        method = 'GWI'
        if method=='GWI':
            e=experiment_regression_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
            e.run()

        elif method=='GPR':
            e=gpr_reference(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
            e.run()
        # y_hat=e.predict_mean(X.cuda()).squeeze()
        # y_hat_q=e.predict_uncertainty(X.cuda()).squeeze()
        # l = (y_hat - 1.96*y_hat_q).cpu().squeeze()
        # u =  (y_hat + 1.96*y_hat_q).cpu().squeeze()
    # elif method=='GP':
    #     e = gp_full_baseline(train_x=X_tr.squeeze(),train_y=y_tr.squeeze(),train_params=training_params)
    #     e.to('cuda:0')
    #     e.train_model()
    #     y_hat,l,u = e.eval_model(X.squeeze().cuda())
    #     y_hat,l,u = y_hat.cpu(),l.cpu(),u.cpu()
    # elif method=='SVGP':
    #     e = gp_svi_baseline(train_x=X_tr,train_y=y_tr,train_params=training_params,VI_params=VI_params)
    #     e.to('cuda:0')
    #     e.train_model()
    #     y_hat,l,u = e.eval_model(X.cuda())
    #     y_hat,l,u = y_hat.cpu(),l.cpu(),u.cpu()

    # sns.scatterplot(X.squeeze(),y.squeeze())
    # ax=sns.lineplot(X.squeeze(),y_hat.cpu())
    # ax.fill_between(X.squeeze(),l,u, color='b', alpha=.5)
    # plt.savefig(f'{method}_{index}.png')

    #FIGURE OUT SCALING ISSUE


