from utils.custom_run import *
nn_params = {
    'layers_x': [10,10],
    'cat_size_list': [],
    'output_dim': 1,
}
VI_params={
    'q_kernel':'r_param_scaling',#'r_param_simple',#'r_param_scaling'
    'p_kernel':'rbf',
    'm_p':0.0,
    'reg':1e-2,
    'r':50,
    'APQ': True,
}
h_space={
    'depth_x':[2],
    'width_x':[10],
    'bs':[2500],
    'lr':[1e-2],
    'm_P':[0.0],
    'sigma':[1e-7],
    'transformation':[torch.nn.Tanh()],
    'm_factor':[1.],
    'parametrize_Z': [False],
    'use_all_m': [False],
    'm_q_choice': ['mlp'],

}
training_params = {

                   'patience': 1000,
                   'device': 'cuda:0',
                   'epochs':1,
                   'lr':1e-2,
                   'model_name':'GWI',
                   'savedir':'regression_test_3',
                   'seed':0,
                   'hyperits':1,
                    'init_its':250
                   }
if __name__ == '__main__':
    # ['boston', 'concrete', 'energy' ,'wine', 'yacht']
    for ds in ['boston', 'concrete', 'energy' ,'wine', 'yacht']:
        training_params['dataset'] = ds
        training_params['fold'] = 0
        for xs_size in [50,100,150,200,250]:
            e = diagonse_eigenvalue_decay_regression(x_S_size=xs_size,hyper_param_space=h_space, VI_params=VI_params, train_params=training_params)
            e.run()
