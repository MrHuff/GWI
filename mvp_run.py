from utils.hyperopt_run import *
from simulate_data.unit_test_data import *
import seaborn as sns
import matplotlib.pyplot as plt
from GP_baseline.gp_baseline_exact import *
from GP_baseline.gp_baseline_vi import *
from sklearn.model_selection import train_test_split
import numpy as np
sns.set()
#TODO: upgrade m_Q wtf haha
nn_params = {
    'layers_x': [10,10],
    'cat_size_list': [],
    'transformation': torch.tanh,
    'output_dim': 1,
}
VI_params={
    'q_kernel':'r_param_scaling',
    'p_kernel':'rbf',
    'sigma':1e-4,
    'm_p':0.0,
    'reg':1e-3,
    'r':50,
    'y_var': 10.0,
    'APQ':True,
    'parametrize_Z':True
}

training_params = {'bs': 900,
                   'patience': 10,
                   'device': 'cuda:0',
                   'epochs':10,
                   'lr':1e-2
                   }

def random_split(X,y):
    X_tr, X_val, y_tr, y_val=train_test_split(X, y, test_size=0.1, random_state=42)
    return X_tr, X_val, y_tr, y_val

def forecast_split(X,y,factor=0.9):
    N,d = X.shape
    tr_ind = int(round(factor*N))

    X_tr=X[:tr_ind]
    y_tr=y[:tr_ind]
    X_val=X[tr_ind:]
    y_val=y[tr_ind:]
    return X_tr, X_val, y_tr, y_val

def remove_random_chunks(X,y,chunks_to_remove=5,total_chunks=20):
    np.random.seed(99)
    x_chunks = torch.chunk(X,chunks=total_chunks,dim=0)
    y_chunks = torch.chunk(y,chunks=total_chunks,dim=0)
    remove_these=np.random.choice([i for i in range(20)],chunks_to_remove)
    x_tr_list = []
    y_tr_list = []
    x_val_list=[]
    y_val_list = []
    for i in range(total_chunks):
        if i in remove_these:
            x_val_list.append(x_chunks[i])
            y_val_list.append(y_chunks[i])
        else:
            x_tr_list.append(x_chunks[i])
            y_tr_list.append(y_chunks[i])
    X_tr = torch.cat(x_tr_list,dim=0)
    y_tr = torch.cat(y_tr_list,dim=0)
    X_val = torch.cat(x_val_list,dim=0)
    y_val = torch.cat(y_val_list,dim=0)
    return X_tr, X_val, y_tr, y_val

# def partial_split

def plot_stuff(X,X_tr,y_tr,X_val,y_val,y_hat,l,u,VI_params,method,index):
    sns.scatterplot(X_tr.squeeze(),y_tr.squeeze())
    sns.scatterplot(X_val.squeeze(),y_val.squeeze(),color='r')
    ax=sns.lineplot(X.squeeze(),y_hat.cpu())
    ax.fill_between(X.squeeze(),l,u, color='b', alpha=.5)
    if method=='GWI':
        str_extra = VI_params['APQ']
        plt.savefig(f'{method}_{index}_APQ={str_extra}.png')
    else:
        plt.savefig(f'{method}_{index}.png')
    plt.clf()

def sim_run(index,method):
    if index==1:
        X,y=sim_sin_curve()
    elif index==2:
        X,y=sim_sin_curve_2()
    elif index==3:
        X,y = sim_sin_curve_3(noise=0.25)
    X_tr, X_val, y_tr, y_val=remove_random_chunks(X,y,chunks_to_remove=5,total_chunks=20)
    # method = 'GWI'
    print(y_tr.std().item())
    VI_params['y_var'] = y_tr.std().item()
    if method=='GWI':
        e=mvp_experiment_object(X=X_tr, Y=y_tr, nn_params=nn_params, VI_params=VI_params, train_params=training_params)
        e.fit()
        y_hat=e.predict_mean(X.cuda()).squeeze()
        y_hat_q=e.predict_uncertainty(X.cuda()).squeeze()
        l = (y_hat - 1.96*y_hat_q).cpu().squeeze()
        u =  (y_hat + 1.96*y_hat_q).cpu().squeeze()

    elif method=='GP':
        e = gp_full_baseline(train_x=X_tr.squeeze(),train_y=y_tr.squeeze(),train_params=training_params)
        e.to('cuda:0')
        e.train_model()
        y_hat,l,u = e.eval_model(X.squeeze().cuda())
        y_hat,l,u = y_hat.cpu(),l.cpu(),u.cpu()

    elif method=='SVGP':
        e = gp_svi_baseline(train_x=X_tr,train_y=y_tr,train_params=training_params,VI_params=VI_params)
        e.to('cuda:0')
        e.train_model()
        y_hat,l,u = e.eval_model(X.cuda())
        y_hat,l,u = y_hat.cpu(),l.cpu(),u.cpu()

    plot_stuff(X,X_tr,y_tr,X_val,y_val,y_hat,l,u,VI_params,method,index)

if __name__ == '__main__':
    torch.random.manual_seed(np.random.randint(0,100000))

    for i in [1,2,3]:
        sim_run(i,'SVGP')

    #FIGURE OUT SCALING ISSUE


