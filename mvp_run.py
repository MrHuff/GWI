from utils.hyperopt_run import *
from simulate_data.unit_test_data import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

nn_params = {
    'layers_x': [8, 8],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': torch.tanh,
    'output_dim': 1,
}
VI_params={
    'm':50,
    'q_kernel':'rbf',
    'p_kernel':'rbf',
    'sigma':1e-4,
    'm_p':0.0,
    'reg':1e-1
}


training_params = {'bs': 900,
                   'patience': 10,
                   'device': 'cuda:0',
                   'epochs':1000,
                   'lr':1e-1
                   }
if __name__ == '__main__':
    X,y=sim_sin_curve()
    tr_ind=900
    X_tr=X[:tr_ind]
    y_tr=y[:tr_ind]
    X_val=X[tr_ind:]
    y_val=y[tr_ind:]
    e=experiment_object(X=X_tr,Y=y_tr,nn_params=nn_params,VI_params=VI_params,train_params=training_params)
    e.fit()
    y_hat=e.predict_mean(X.cuda()).squeeze()
    y_hat_q=e.predict_uncertainty(X.cuda()).squeeze()

    sns.scatterplot(X.squeeze(),y.squeeze())
    ax=sns.lineplot(X.squeeze(),y_hat.cpu())
    ax.fill_between(X.squeeze(), (y_hat - y_hat_q).cpu().squeeze(), (y_hat + y_hat_q).cpu().squeeze(), color='b', alpha=.5)

    plt.show()





