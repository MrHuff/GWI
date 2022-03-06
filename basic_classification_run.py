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
    'm':100,
    'q_kernel':'r_param',
    'p_kernel':'rbf',
    'sigma':1e-4,
    'm_p':0.0,
    'reg':1e-2,
    'r':50,
    'y_var': 10.0,
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
                    'dataset':'CIFAR10'
                   }

h_space={
    'depth_x':[3],
    'width_x':[8,16,32],
    'bs':[1000],
    'lr':[1e-2],
    'm_P':[0.5],
    'sigma':[1e-3],
    'transformation':[torch.relu],
    'depth_fc':[1]
}
if __name__ == '__main__':

    torch.random.manual_seed(np.random.randint(0,100000))
    e=experiment_classification_object(hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
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


