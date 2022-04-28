import numpy as np
from utils.hyperopt_run import *

class regression_object(experiment_regression_object):
    def __init__(self,X,Y,hyper_param_space,VI_params,train_params,device='cuda:0'):
        super(regression_object, self).__init__(hyper_param_space,VI_params,train_params,device)
        self.X = X
        self.Y = Y
        self.ds=general_custom_dataset(X,Y,[])
        self.empirical_sigma = Y.std().item()

    def __call__(self, parameters_in):
        model_copy = dill.dumps(self.trials)
        pickle.dump(model_copy,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))
        self.dataloader = custom_dataloader(self.ds,parameters_in['bs'])
        self.n, self.d = self.dataloader.dataset.train_X.shape
        self.X = self.dataloader.dataset.train_X
        self.Y = self.dataloader.dataset.train_y
        print(self.X.shape)
        if parameters_in['use_all_m']:
            self.m = self.n
        else:
            self.m = int(round(self.X.shape[0] ** 0.5) * parameters_in['m_factor'])
        parameters_in['m'] = self.m
        parameters_in['init_its'] = self.train_params['init_its']
        self.sigma = parameters_in['sigma']
        mean_y_train = self.Y.mean().item()
        nn_params = {
            'd_in_x': self.d,
            'cat_size_list': [],
            'output_dim': 1,
            'transformation': parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']] * parameters_in['depth_x'],
        }
        if self.n > parameters_in['m']:
            z_mask = torch.randperm(self.n)[:parameters_in['m']]
            self.Z = self.X[z_mask, :]
            self.Y_Z = self.Y[z_mask, :]
            self.X_hat = self.X[torch.randperm(self.n)[:parameters_in['m']], :]
        else:
            self.Z = copy.deepcopy(self.X)
            self.Y_Z = copy.deepcopy(self.Y)
            self.X_hat = copy.deepcopy(self.X)
        self.k = self.get_kernels(self.VI_params['p_kernel'], parameters_in)
        self.r = self.get_kernels(self.VI_params['q_kernel'], parameters_in)
        if parameters_in['m_q_choice'] == 'kernel_sum':
            nn_params['k'] = self.k
            nn_params['Z'] = self.Z
            self.m_q = kernel_feature_map_regression(**nn_params).to(self.device)
        elif parameters_in['m_q_choice'] == 'mlp':
            self.m_q = feature_map(**nn_params).to(self.device)
        elif parameters_in['m_q_choice'] == 'krr':
            self.m_q = KRR_mean(r=self.r)

        self.vi_obj = GWI(
            N=self.X.shape[0],
            m_q=self.m_q,
            m_p=mean_y_train * parameters_in['m_P'],
            r=self.r,
            sigma=parameters_in['sigma'],
            reg=self.VI_params['reg'],
            APQ=self.VI_params['APQ'],
            empirical_sigma=self.empirical_sigma
        ).to(self.device)
        self.lr = parameters_in['lr']
        val_loss, test_loss, valr2, testr2, val_rsme, test_rsme, T = self.fit()
        self.global_hyperit += 1
        return {
            'loss': val_loss,
            'status': STATUS_OK,
            'test_loss': test_loss,
            'val_r2': valr2,
            'test_r2': testr2,
            'net_params': nn_params,
            'val_rsme': val_rsme,
            'test_rsme': test_rsme,
            'T': T
        }

class diagonse_eigenvalue_decay_regression(experiment_regression_object):
    def __init__(self,hyper_param_space,VI_params,train_params,x_S_size=50,device='cuda:0'):
        super(diagonse_eigenvalue_decay_regression, self).__init__(hyper_param_space,VI_params,train_params,device)
        self.x_S_size = x_S_size
        self.ds_string = self.train_params['dataset']
        self.fold = self.train_params['fold']

    def __call__(self,parameters_in):
        print(parameters_in)
        model_copy = dill.dumps(self.trials)
        pickle.dump(model_copy,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))
        self.dataloader,self.ds = get_regression_dataloader(dataset=self.train_params['dataset'],fold=self.train_params['fold'],bs=parameters_in['bs'])
        self.n,self.d = self.dataloader.dataset.train_X.shape
        self.X=self.dataloader.dataset.train_X
        self.Y=self.dataloader.dataset.train_y
        print(self.X.shape)
        if parameters_in['use_all_m']:
            self.m=self.n
        else:
            self.m=int(round(self.X.shape[0]**0.5)*parameters_in['m_factor'])
        parameters_in['m'] = self.m
        parameters_in['init_its'] = self.train_params['init_its']
        self.sigma=parameters_in['sigma']
        mean_y_train = self.Y.mean().item()
        nn_params = {
            'd_in_x' : self.d,
            'cat_size_list': [],
            'output_dim' : 1,
            'transformation':parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']]*parameters_in['depth_x'],
        }
        if self.n>parameters_in['m']:
            z_mask=torch.randperm(self.n)[:parameters_in['m']]
            self.Z =  self.X[z_mask, :]
            self.Y_Z= self.Y[z_mask,:]
            self.X_hat =  self.X[torch.randperm(self.n)[:parameters_in['m']], :]
        else:
            self.Z= copy.deepcopy(self.X)
            self.Y_Z= copy.deepcopy(self.Y)
            self.X_hat=copy.deepcopy(self.X)
        self.k=self.get_kernels(self.VI_params['p_kernel'],parameters_in)
        self.r=self.get_kernels(self.VI_params['q_kernel'],parameters_in)
        if parameters_in['m_q_choice']=='kernel_sum':
            nn_params['k'] = self.k
            nn_params['Z'] = self.Z
            self.m_q = kernel_feature_map_regression(**nn_params).to(self.device)
        elif parameters_in['m_q_choice']=='mlp':
            self.m_q = feature_map(**nn_params).to(self.device)
        elif parameters_in['m_q_choice']=='krr':
            self.m_q = KRR_mean(r=self.r)

        self.vi_obj=GWI(
                        N=self.X.shape[0],
                        m_q=self.m_q,
                        m_p=mean_y_train*parameters_in['m_P'],
                        r=self.r,
                        sigma=parameters_in['sigma'],
                        reg=self.VI_params['reg'],
                        APQ=self.VI_params['APQ'],
                    empirical_sigma=self.ds.empirical_sigma,
            x_s=self.x_S_size
        ).to(self.device)
        self.lr = parameters_in['lr']
        val_loss,test_loss,valr2,testr2,val_rsme,test_rsme,T=self.fit()
        self.global_hyperit+=1
        return  {
                'loss': val_loss,
                'status': STATUS_OK,
                'test_loss': test_loss,
                'val_r2':valr2,
                'test_r2':testr2,
                 'net_params':nn_params,
                'val_rsme': val_rsme,
                'test_rsme': test_rsme,
                'T':T
                }
    def train_loop(self,opt):
        self.vi_obj.train()
        self.dataloader.dataset.set('train')
        pbar= tqdm.tqdm(self.dataloader)
        for i,(X,x_cat,y) in enumerate(pbar):
            X=X.to(self.device)
            y=y.to(self.device)
            # Investigate X_s, eigen value decay argument

            z_mask=torch.randperm(self.n)[:self.x_S_size]
            Z_prime = self.X[z_mask, :].to(self.device)
            log_loss,D=self.vi_obj.get_loss(y,X,Z_prime)
            self.eigen_values = self.vi_obj.get_APQ_diagnose_xs(X,Z_prime)
            # print(self.vi_obj.r.k.base_kernel.lengthscale)
            pbar.set_description(f"D: {D.item()} log_loss: {log_loss.item() }")
            tot_loss = D + log_loss
            opt.zero_grad()
            tot_loss.backward()
            opt.step()
        self.eigen_values = np.sort(self.eigen_values)
        sns.barplot(x=np.arange(1,len(self.eigen_values)+1), y=self.eigen_values[::-1])
        plt.savefig(self.save_path + f'eig_val_decay_{self.x_S_size}_{self.ds_string}_{self.fold}.png')
        plt.clf()




