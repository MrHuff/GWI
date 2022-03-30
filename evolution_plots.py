import os.path

from utils.hyperopt_run import *
from simulate_data.unit_test_data import *
import seaborn as sns
import matplotlib.pyplot as plt
from GP_baseline.gp_baseline_exact import *
from GP_baseline.gp_baseline_vi import *
from sklearn.model_selection import train_test_split
import numpy as np
import imageio

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
                   'patience': 1000,
                   'device': 'cuda:0',
                   'epochs':2000,
                   'lr':1e-2
                   }

def generate_gif(filenames,dir,gif_name):
    with imageio.get_writer(f'{dir}/{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

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

def plot_stuff_2(sigma,d_val,method,index,dir,epoch):
    if not os.path.exists(dir):
        os.makedirs(dir)
    sns.heatmap(sigma)
    fname=f'{dir}/{method}_{index}_{epoch}.png'
    plt.title(f'Epoch: {epoch}, d: {d_val}')
    plt.savefig(fname)
    plt.close()
    plt.clf()
    return fname

def plot_stuff(X_inducing,Y_inducing,X,X_tr,y_tr,X_val,y_val,y_hat,l,u,method,index,dir,epoch):
    if not os.path.exists(dir):
        os.makedirs(dir)

    sns.scatterplot(X_tr.squeeze(),y_tr.squeeze(),alpha=0.5)
    sns.scatterplot(X_inducing.squeeze(),Y_inducing.squeeze(),color='y')
    sns.scatterplot(X_val.squeeze(),y_val.squeeze(),color='r',alpha=0.5)
    ax=sns.lineplot(X.squeeze(),y_hat)
    ax.fill_between(X.squeeze(),l,u, color='b', alpha=.5)
    fname=f'{dir}/{method}_{index}_{epoch}.png'
    plt.title(f'Epoch: {epoch}')
    plt.savefig(fname)
    plt.close()
    plt.clf()
    return fname

def get_u_l(y_hat,y_hat_q):
    l = (y_hat - 1.96 * y_hat_q)
    u = (y_hat + 1.96 * y_hat_q)
    return u,l

def sim_run(index,method):
    p_z=VI_params['parametrize_Z']
    dir_name = f'gwi_gif_{index}_param_z={p_z}'
    dir_name_2 = f'heatmap_gwi_gif_{index}_param_z={p_z}'
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
    e=mvp_experiment_object(X=X_tr, Y=y_tr, nn_params=nn_params, VI_params=VI_params, train_params=training_params)
    e.fit(X.cuda())
    filenames=[]
    x_inducing=e.Z.cpu().numpy()
    y_inducing=e.Y_Z.cpu().numpy()
    for i,(a,b) in enumerate(zip(e.preds,e.vars)):
        if i%10==0:
            l,u=get_u_l(a,b)
            fname=plot_stuff(X_inducing=x_inducing,Y_inducing=y_inducing,X=X,X_tr=X_tr,y_tr=y_tr,X_val=X_val,y_val=y_val,y_hat=a,l=l,u=u,method=method,index=index,dir=dir_name,epoch=i)
            filenames.append(fname)
    generate_gif(filenames,dir_name,f'line_plot_{index}')
    filenames=[]
    for i,(a,b) in enumerate(zip(e.d_vals,e.mat_list)):
        if i%10==0:
            fname = plot_stuff_2(b,a,method,index,dir_name_2,i)
            filenames.append(fname)
    generate_gif(filenames,dir_name_2,f'heatmap_{index}')

if __name__ == '__main__':
    torch.random.manual_seed(np.random.randint(0,100000))
    for i in [1,2,3]:
        sim_run(i,'GWI')

    #FIGURE OUT SCALING ISSUE


