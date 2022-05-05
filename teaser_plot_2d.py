import torch
from utils.utils_2d_plot import *
from utils.custom_run import *
import shutil
nn_params = {
    'layers_x': [50],
    'cat_size_list': [],
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
    'width_x':[25],
    'bs':[100],
    'lr':[1e-2],
    'm_P':[0.0],
    'sigma':[1e-7],
    'transformation':[torch.nn.Tanh()],
    'm_factor':[1.],
    'parametrize_Z': [False],
    'use_all_m': [True],
    'm_q_choice': ['mlp'],
    'x_s':[50],

}
training_params = {

                   'patience': 1000,
                   'device': 'cuda:0',
                   'epochs':1000,
                   'lr':1e-3,
                   'model_name':'GWI',
                   'savedir':'2d_plot_nn_kernel',
                   'seed':0,
                   'hyperits':1,
                    'init_its':100,
                    'dataset':'any',
                    'fold':0
                   }
if __name__ == '__main__':
    plot=True
    save=True
    dataset='origin'
    inference = 'GWI'
    layers = 1
    X,y = load_X_y(dataset)
    savedir = training_params['savedir']
    dirname = training_params['savedir']
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    e = regression_object(X=torch.from_numpy(X).float(),Y=torch.from_numpy(y).float(),
        hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
    e.run()

    contour_std, slice_mean, slice_std = get_2d_pred_GWI(e, dataset)
    os.makedirs(savedir, exist_ok=True)
    if plot:
        figpath = Path(savedir, f'{layers}HL_{inference}.pdf')
        make_2d_plot(contour_std, slice_mean, slice_std, dataset, figpath)
    if save:
        pickle_path = Path(savedir,
                           f'{layers}HL_{inference}.pkl')
        save_2d(contour_std, slice_mean, slice_std, pickle_path)