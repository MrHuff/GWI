import os.path

from utils.custom_run import *
from simulate_data.unit_test_data import *
import seaborn as sns
import matplotlib.pyplot as plt
from GP_baseline.gp_baseline_exact import *
from GP_baseline.gp_baseline_vi import *
from sklearn.model_selection import train_test_split
import numpy as np
import imageio
import shutil
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
sns.set()
print(sns.color_palette("tab10")
)
#TODO: upgrade m_Q wtf haha
nn_params = {
    'layers_x': [10,10],
    'cat_size_list': [],
    'transformation': torch.tanh,
    'output_dim': 1,
}
VI_params={
    'q_kernel':'r_param_scaling',#'nn_kernel',#'r_param_scaling',#'r_param_simple'
    'p_kernel':'rbf',
    'm_p':0.0,
    'reg':1e-2,
    'r':50,
    'APQ': True,
}
h_space={
    'depth_x':[2],
    'width_x':[50],
    'bs':[1000],
    'lr':[1e-2],
    'm_P':[1.0],
    'sigma':[1e-3],
    'transformation':[torch.nn.Tanh()],
    'm_factor':[1.],
    'parametrize_Z': [False],
    'use_all_m': [False],
    'm_q_choice': ['mlp'],
    'x_s':[200],
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
                    'init_its':250,
                    'dataset':'None',
                    'fold': 0,
                    'bs': 1000,

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

def plot_stuff(equation,X_inducing,Y_inducing,X,X_tr,y_tr,X_val,y_val,y_hat,l,u,method,index,dir,epoch):
    if not os.path.exists(dir):
        os.makedirs(dir)

    sns.scatterplot(X_tr.squeeze(),y_tr.squeeze(),alpha=0.35,color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
    try:
        sns.scatterplot(X_inducing.squeeze(),Y_inducing.squeeze(),color= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),alpha=1.0)
    except Exception as e:
        pass
    sns.scatterplot(X_val.squeeze(),y_val.squeeze(),color=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),alpha=0.35)
    ax=sns.lineplot(X.squeeze(),y_hat.squeeze())
    ax.fill_between(X.squeeze(),l,u, color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), alpha=.15)
    fname=f'{dir}/{method}_{index}_{epoch}.png'
    plt.title(fr'${equation}$',fontsize=15)
    plt.xlabel(r"$x$",fontsize=18)
    plt.ylabel(r"$y(x)$",fontsize=18)
    plt.savefig(fname,bbox_inches = 'tight',
            pad_inches = 0.05)
    plt.close()
    plt.clf()
    return fname

def get_u_l(y_hat,y_hat_q):
    l = (y_hat - 1.96 * y_hat_q)
    u = (y_hat + 1.96 * y_hat_q)
    return u,l

def sim_run(index,method):
    p_z=False
    dir_name = f'gwi_gif_{index}_param_z={p_z}'
    dir_name_2 = f'heatmap_gwi_gif_{index}_param_z={p_z}'
    if index==1:
        X,y=sim_sin_curve()
        equation = r'y(x)=\sin(x)+0.1x+\epsilon'
    elif index==2:
        X,y=sim_sin_curve_2()
        equation = r'y(x)=\sin(x)+0.1x^2+\epsilon'
    elif index==3:
        X,y = sim_sin_curve_3(noise=0.25)
        equation = r'y(x)=\sin(3\pi x)+0.3\cos(9\pi x)+0.5\sin(7\pi x)+ \epsilon'

    X_tr, X_val, y_tr, y_val=remove_random_chunks(X,y,chunks_to_remove=8,total_chunks=20)
    # X_tr, X_val, y_tr, y_val=forecast_split(X,y,factor=0.75)
    # method = 'GWI'
    print(y_tr.std().item())
    VI_params['y_var'] = y_tr.std().item()
    dirname = f'regression_test_1_False'
    training_params['savedir'] = dirname
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    if method=='GWI':
        e=regression_object(X=X_tr.float(),Y=y_tr.float(),
            hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
            # mvp_experiment_object(X=X_tr, Y=y_tr, nn_params=nn_params, VI_params=VI_params, train_params=training_params)
        e.run()
        x_inducing=e.Z.cpu().numpy()
        y_inducing=e.Y_Z.cpu().numpy()
        y_hat,b=e.pred_mean_std(X.cuda())
        l, u = get_u_l(y_hat.squeeze(), b.squeeze())
    elif method=='GP':
        e = gp_full_baseline(train_x=X_tr.squeeze(),train_y=y_tr.squeeze(),train_params=training_params)
        e.to('cuda:0')
        e.train_model()
        y_hat,l,u = e.eval_model(X.squeeze().cuda())
        y_hat,l,u = y_hat.cpu(),l.cpu(),u.cpu()
        x_inducing=X.cpu().squeeze()
        y_inducing=y.cpu().squeeze()

    elif method=='SVGP':
        e = gp_svi_baseline(train_x=X_tr,train_y=y_tr,train_params=training_params,VI_params=VI_params)
        e.to('cuda:0')
        e.train_model()
        x_inducing=e.inducing_points.cpu().numpy()
        y_inducing=e.inducing_points_y.cpu().numpy()
        y_hat,l,u = e.eval_model(X.cuda())
        y_hat,l,u = y_hat.cpu(),l.cpu(),u.cpu()

    fname = plot_stuff(equation=equation,X_inducing=x_inducing, Y_inducing=y_inducing, X=X, X_tr=X_tr.cpu(), y_tr=y_tr.cpu(), X_val=X_val.cpu(),
                       y_val=y_val.cpu(), y_hat=y_hat.cpu(), l=l.cpu(), u=u.cpu(), method=method, index=index, dir=dir_name, epoch=1000)



    # filenames=[]
    # for i,(a,b) in enumerate(zip(e.preds,e.vars)):
    #     if i%10==0:
    #         l,u=get_u_l(a,b)
    #         fname=plot_stuff(X_inducing=x_inducing,Y_inducing=y_inducing,X=X,X_tr=X_tr,y_tr=y_tr,X_val=X_val,y_val=y_val,y_hat=a,l=l,u=u,method=method,index=index,dir=dir_name,epoch=i)
    #         filenames.append(fname)
    # generate_gif(filenames,dir_name,f'line_plot_{index}')
    # filenames=[]
    # for i,(a,b) in enumerate(zip(e.d_vals,e.mat_list)):
    #     if i%10==0:
    #         fname = plot_stuff_2(b,a,method,index,dir_name_2,i)
    #         filenames.append(fname)
    # generate_gif(filenames,dir_name_2,f'heatmap_{index}')

if __name__ == '__main__':
    torch.random.manual_seed(np.random.randint(0,100000))
    # for i in [1,2,3]:
    # for i in [1]:
    # for i in [1,2,3]:
    for i in [3]:
        sim_run(i,'GWI')
        # sim_run(i,'GP')
        # sim_run(i,'SVGP')

    #FIGURE OUT SCALING ISSUE


