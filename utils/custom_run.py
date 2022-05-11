import copy

import numpy as np
import torch

from utils.hyperopt_run import *
import gpytorch
class regression_object(experiment_regression_object):
    def __init__(self,X,Y,hyper_param_space,VI_params,train_params,device='cuda:0'):
        super(regression_object, self).__init__(hyper_param_space,VI_params,train_params,device)
        self.X = X
        self.Y = Y
        self.ds=general_custom_dataset(X,Y,[])
        self.empirical_sigma = Y.std().item()
    # def train_loop(self,opt,T=None,mode='train'):
    #     self.vi_obj.train()
    #     self.dataloader.dataset.set(mode)
    #     pbar= tqdm.tqdm(self.dataloader)
    #     for i,(X,x_cat,y) in enumerate(pbar):
    #         X=X.to(self.device)
    #         y=y.to(self.device)
    #         # Investigate X_s, eigen value decay argument
    #
    #         z_mask=torch.randperm(self.n)[:self.vi_obj.x_s]
    #         Z_prime = self.X[z_mask, :].to(self.device)
    #         log_loss,D=self.vi_obj.get_loss(y,X,Z_prime,T)
    #         # print(self.vi_obj.r.k.base_kernel.lengthscale)
    #         pbar.set_description(f"D: {D.item()} log_loss: {log_loss.item() }")
    #         tot_loss = 0.1*D + log_loss
    #         opt.zero_grad()
    #         tot_loss.backward()
    #         opt.step()

    def __call__(self, parameters_in):
        self.vi_obj_copy = dill.dumps(self.trials)
        pickle.dump(self.vi_obj_copy,
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
        mean_y_train = self.Y.mean().item()
        nn_params = {
            'd_in_x': self.d,
            'cat_size_list': [],
            'output_dim': 1,
            'transformation': parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']] * parameters_in['depth_x'],
        }
        self.nn_params = copy.deepcopy(nn_params)
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
        if parameters_in['m_q_choice'] == 'kernel_sum':
            nn_params['k'] = self.k
            nn_params['Z'] = self.Z
            self.m_q = kernel_feature_map_regression(**nn_params).to(self.device)
        elif parameters_in['m_q_choice'] == 'mlp':
            self.m_q = feature_map(**nn_params).to(self.device)
        elif parameters_in['m_q_choice'] == 'krr':
            self.r = self.get_kernels(self.VI_params['q_kernel'], parameters_in)
            self.m_q = KRR_mean(r=self.r)
        self.r = self.get_kernels(self.VI_params['q_kernel'], parameters_in)

        self.vi_obj = GWI(
            N=self.X.shape[0],
            m_q=self.m_q,
            m_p=mean_y_train * parameters_in['m_P'],
            r=self.r,
            sigma=self.sigma,
            reg=self.VI_params['reg'],
            APQ=self.VI_params['APQ'],
            empirical_sigma=self.empirical_sigma,
            x_s=parameters_in['x_s']
        ).to(self.device)
        self.lr = parameters_in['lr']
        self.fit()
        val_loss,test_loss,valr2,testr2,val_rsme,test_rsme,T= self.final_evaluation()
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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class gpr_reference(experiment_regression_object):
    def calc_nll(self,y,val_output,noise):
        pred = val_output.mean
        rXX = val_output.variance + noise.item()
        vec=(y-pred)**2
        a = (0.5 * torch.log(rXX)).sum()
        b = (0.5 * vec / rXX).sum()
        return ((a + b) + self.const * pred.shape[0])/pred.shape[0]
    def helper_plot(self,Y,Y_pred,mse_per_ob,all_vars,mode):
        df = pd.DataFrame(np.stack([Y, Y_pred, mse_per_ob, all_vars], axis=1), columns=['Y', 'Y_pred', 'mse', 'var'])
        sns.lineplot(data=df, x='Y', y='Y')
        sns.scatterplot(data=df, x="Y", y="Y_pred")
        plt.savefig(self.save_path + f'{mode}_y_plots_{self.global_hyperit}.png')
        plt.clf()
        # sns.lineplot(data=df,x='mse',y='mse')
        sns.scatterplot(data=df, x="var", y="mse")
        plt.savefig(self.save_path + f'{mode}_mse_var_{self.global_hyperit}.png')
        plt.clf()
    def create_NLL_plot(self):
        self.vi_obj.eval()
        noise = self.vi_obj.likelihood.noise
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_output = self.vi_obj(self.val_X)
            test_output = self.vi_obj(self.test_X)

            pred_val = val_output.mean
            rXX_val = val_output.variance + noise.item()
            mse_val = (self.val_Y.squeeze().cpu().numpy()-pred_val.cpu().numpy())**2

            pred_test = test_output.mean
            rXX_test = test_output.variance + noise.item()
            mse_test = (self.test_Y.squeeze().cpu().numpy()-pred_test.cpu().numpy())**2

        self.helper_plot(Y=self.val_Y.squeeze().cpu().numpy(),
                         Y_pred=pred_val.cpu().numpy(),
                         mse_per_ob=mse_val,all_vars=rXX_val.squeeze().cpu().numpy(),mode=
                         'val')
        self.helper_plot(Y=self.test_Y.squeeze().cpu().numpy(),
                         Y_pred=pred_test.cpu().numpy(),
                         mse_per_ob=mse_test,all_vars=rXX_test.squeeze().cpu().numpy(),mode=
                         'test')

    def __call__(self, parameters_in):
        print(parameters_in)
        self.vi_obj_copy = dill.dumps(self.trials)
        pickle.dump(self.vi_obj_copy,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))
        self.dataloader, self.ds = get_regression_dataloader(dataset=self.train_params['dataset'],
                                                             fold=self.train_params['fold'], bs=parameters_in['bs'])
        self.n, self.d = self.dataloader.dataset.train_X.shape
        self.X = self.dataloader.dataset.train_X.to(self.device)
        self.Y = self.dataloader.dataset.train_y.to(self.device)
        self.val_X = self.dataloader.dataset.val_X.to(self.device)
        self.val_Y = self.dataloader.dataset.val_y.to(self.device)
        self.test_X = self.dataloader.dataset.test_X.to(self.device)
        self.test_Y = self.dataloader.dataset.test_y.to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.vi_obj = ExactGPModel(self.X, self.Y.squeeze(), likelihood).to(self.device)
        optimizer = torch.optim.Adam(self.vi_obj.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.vi_obj)
        best_loss = np.inf
        self.log_empirical_sigma = np.log(self.ds.empirical_sigma)
        self.const = (0.5 * log2pi + self.log_empirical_sigma).item()


        for i in tqdm.tqdm(range(1000)):
            self.vi_obj.train()
            likelihood.train()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from self.vi_obj
            output = self.vi_obj(self.X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.Y.squeeze())
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.vi_obj.eval()
                likelihood.eval()
                val_output = self.vi_obj(self.val_X)
                val_NLL = self.calc_nll(self.val_Y.squeeze(),val_output,self.vi_obj.likelihood.noise)
                print(val_NLL)
                if val_NLL.item()<best_loss:
                    best_loss=val_NLL.item()
                    self.dump_model(self.global_hyperit)
        self.load_model(self.global_hyperit)
        self.vi_obj.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_output = self.vi_obj(self.val_X)
            val_NLL = self.calc_nll(self.val_Y.squeeze(), val_output,self.vi_obj.likelihood.noise)

            test_output = self.vi_obj(self.test_X)
            test_NLL = self.calc_nll(self.test_Y.squeeze(), test_output,self.vi_obj.likelihood.noise)

        val_rsme=cuda_rmse(val_output.mean, self.val_Y.squeeze())
        test_rsme=cuda_rmse(test_output.mean, self.test_Y.squeeze())
        valr2=cuda_r2(val_output.mean, self.val_Y.squeeze())
        testr2=cuda_r2(test_output.mean, self.test_Y.squeeze())
        self.global_hyperit += 1
        self.create_NLL_plot()
        print(self.vi_obj.likelihood.noise.item())
        loss_dict = {
            'loss': val_NLL.item(),
            'status': STATUS_OK,
            'test_loss': test_NLL.item(),
            'val_r2': valr2,
            'test_r2': testr2,
            'net_params': None,
            'val_rsme': val_rsme,
            'test_rsme': test_rsme,
            'T': None
        }
        print(loss_dict)
        return loss_dict
class diagonse_eigenvalue_decay_regression(experiment_regression_object):
    def __init__(self,hyper_param_space,VI_params,train_params,x_S_size=50,device='cuda:0'):
        super(diagonse_eigenvalue_decay_regression, self).__init__(hyper_param_space,VI_params,train_params,device)
        self.x_S_size = x_S_size
        self.ds_string = self.train_params['dataset']
        self.fold = self.train_params['fold']

    def __call__(self,parameters_in):
        print(parameters_in)
        self.vi_obj_copy = dill.dumps(self.trials)
        pickle.dump(self.vi_obj_copy,
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




