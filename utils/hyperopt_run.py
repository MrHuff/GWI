import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gwi_VI.mean_models import *
from gwi_VI.models import *
from gpytorch.kernels import RBFKernel,LinearKernel,CosineKernel,ScaleKernel
from utils.dataloaders import *
import seaborn as sns
from utils.regression_dataloaders import *
import tqdm
from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL,rand
import pickle
from utils.classification_dataloaders import *
import dill
import scipy
from sklearn.metrics import roc_auc_score
import matplotlib
from utils.sa import  *
sns.set()
matplotlib.use('agg')

def cuda_r2(y_pred,y):
    res = (y_pred-y)**2
    return (1. - res.mean()/y.var()).item()
def cuda_rmse(y_pred,y):
    res = (y_pred-y)**2
    return (torch.mean(res)**0.5).item()

def covar_dist(x1,x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    # Zero out negative values
    res.clamp_min_(0)

    # res  = torch.cdist(x1,x2,p=2)
    # Zero out negative values
    # res.clamp_min_(0)
    return res

def get_median_ls( X):
    if X.shape[0]>2500:
        X_in = X[torch.randperm(X.shape[0])[:2500]]
    else:
        X_in = X

    with torch.no_grad():
        d = covar_dist(x1=X_in, x2=X_in)
        ret = torch.sqrt(torch.median(d[d >= 0]))  # print this value, should be increasing with d
        if ret.item() == 0:
            ret = torch.tensor(1.0)
        return ret


class mvp_experiment_object():
    def __init__(self,X,Y,nn_params,VI_params,train_params):
        self.X = X
        self.Y = Y
        self.n,self.d = X.shape
        self.device=train_params['device']

        self.VI_params = VI_params
        self.train_params = train_params

        nn_params['d_in_x']=self.X.shape[1]
        self.m_q = feature_map(**nn_params).to(self.device)
        self.m_p = self.VI_params['m_p']
        self.VI_params['m'] = int(round(self.n**0.5))
        if self.n>self.VI_params['m']:
            z_mask=torch.randperm(self.n)[:self.VI_params['m']]
            self.Z =  self.X[z_mask, :]
            self.Y_Z= self.Y[z_mask,:]
            self.X_hat =  self.X[torch.randperm(self.n)[:self.VI_params['m']], :]
        else:
            self.Z=self.X
        self.k=self.get_kernels(self.VI_params['p_kernel'],False)
        self.k=self.k.to(self.device)
        self.r=self.get_kernels(self.VI_params['q_kernel'],True)
        self.vi_obj=GWI(N=self.X.shape[0],
                        m_q=self.m_q,
                        m_p=self.m_p,
                        r=self.r,
                        sigma=self.VI_params['sigma'],
                        reg=self.VI_params['reg'],
                        APQ=self.VI_params['APQ'],
                        empirical_sigma=1.0
                        ).to(self.device)
        dataset=general_custom_dataset(X,Y,nn_params['cat_size_list'])
        self.dataloader=custom_dataloader(dataset,train_params['bs'])


    def get_kernels(self,string,is_q=False):
        if string=='rbf':
            l = RBFKernel(ard_num_dims=self.Z.shape[1])
            # ls=get_median_ls(self.X).to(self.device)
            # l._set_lengthscale(ls)
            k = ScaleKernel(l)
            k._set_outputscale(self.VI_params['y_var'])
            ls_obj=ls_init(k,self.Y_Z,self.Z,sigma=self.VI_params['sigma']).to(self.device)
            ls_obj.pre_train()
            print(l.lengthscale,k.outputscale)

            k.requires_grad_(False)
        elif string == 'r_param_scaling':
            k = r_param_cholesky_scaling(k=self.k, Z=self.Z, X=self.X_hat, sigma=self.k.outputscale.item(),parametrize_Z=self.VI_params['parametrize_Z']).to(self.device)
            k.init_L()
            # k.requires_grad_(True)
        # if is_q:
        #
        # else:
        return k

    def train_loop_m_q(self,opt):
        self.dataloader.dataset.set('train')
        self.vi_obj.train()
        loss = torch.nn.MSELoss()
        for i,(X,x_cat,y) in enumerate(tqdm.tqdm(self.dataloader)):
            X=X.to(self.device)
            y=y.to(self.device)
            y_pred=self.vi_obj.mean_forward(X)
            tot_loss = loss(y_pred,y)
            print(tot_loss.item())
            opt.zero_grad()
            tot_loss.backward()
            opt.step()

    def train_loop(self,opt):
        self.dataloader.dataset.set('train')
        self.vi_obj.train()
        self.m_q.requires_grad_(False)
        D_list = []
        for i,(X,x_cat,y) in enumerate(tqdm.tqdm(self.dataloader)):
            X=X.to(self.device)
            y=y.to(self.device)
            z_mask=torch.randperm(self.n)[:self.VI_params['m']]
            Z_prime = self.X[z_mask, :].to(self.device)
            log_loss,D=self.vi_obj.get_loss(y,X,Z_prime)
            print('D: ',D.item())
            print('log_loss: ',log_loss.item())
            tot_loss = D + log_loss
            opt.zero_grad()
            tot_loss.backward()
            opt.step()
            D_list.append(D.item())
        return min(D_list)

    def fit(self,X=None):
        opt=torch.optim.Adam(self.vi_obj.parameters(),lr=self.train_params['lr'])
        self.preds=[self.predict_mean(X).cpu().numpy()]
        self.vars=[self.predict_uncertainty(X).cpu().numpy()]
        self.d_vals =[]
        self.mat_list = []
        for i in range(2*self.train_params['epochs']):
            self.train_loop_m_q(opt)
        for i in range(self.train_params['epochs']):
            d_min=self.train_loop(opt)
            self.d_vals.append(d_min)
            sigma=self.vi_obj.r.get_sigma_debug()
            # print(sigma)
            self.mat_list.append(sigma.cpu().numpy())
            if X is not None:
                self.preds.append(self.predict_mean(X).cpu().numpy())
                self.vars.append(self.predict_uncertainty(X).cpu().numpy())

    def predict_mean(self,x_test):
        self.vi_obj.eval()
        return self.vi_obj.mean_pred(x_test).squeeze()

    def predict_uncertainty(self, x_test):
        self.vi_obj.eval()
        return self.vi_obj.posterior_variance(x_test).squeeze()

class experiment_regression_object():
    def __init__(self,hyper_param_space,VI_params,train_params,device='cuda:0'):
        self.train_params=train_params
        self.VI_params = VI_params
        self.device = device
        self.hyperopt_params = ['transformation', 'depth_x', 'width_x', 'bs', 'lr','m_P','sigma','m_factor','parametrize_Z','use_all_m']
        self.get_hyperparameterspace(hyper_param_space)
        self.generate_save_path()
        self.dataloader,self.ds = get_regression_dataloader(dataset=self.train_params['dataset'],fold=self.train_params['fold'],bs=100)
        self.global_hyperit=0

    def generate_save_path(self):
        model_name = self.train_params['model_name']
        savedir = self.train_params['savedir']
        dataset = self.train_params['dataset']
        fold = self.train_params['fold']
        seed = self.train_params['seed']
        self.save_path = f'{savedir}/{dataset}_seed={seed}_fold_idx={fold}_model={model_name}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def dump_model(self,hyperit):
        model_copy = dill.dumps(self.vi_obj)
        torch.save(model_copy, self.save_path+f'best_model_{hyperit}.pt')

    def load_model(self,hyperit):
        model_copy=torch.load(self.save_path+f'best_model_{hyperit}.pt')
        self.vi_obj=dill.loads(model_copy)

    def __call__(self,parameters_in):
        self.dataloader.batch_size=parameters_in['bs']
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
        if self.train_params['m_q_choice']=='kernel_sum':
            nn_params['k'] = self.k
            nn_params['Z'] = self.Z
            self.m_q = kernel_feature_map_regression(**nn_params).to(self.device)
        elif self.train_params['m_q_choice']=='mlp':
            self.m_q = feature_map(**nn_params).to(self.device)
        self.k=self.get_kernels(self.VI_params['p_kernel'],parameters_in)
        self.r=self.get_kernels(self.VI_params['q_kernel'],parameters_in)
        self.vi_obj=GWI(
                        N=self.X.shape[0],
                        m_q=self.m_q,
                        m_p=mean_y_train*parameters_in['m_P'],
                        r=self.r,
                        sigma=parameters_in['sigma'],
                        reg=self.VI_params['reg'],
                        APQ=self.VI_params['APQ'],
                    empirical_sigma=self.ds.empirical_sigma

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

    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def get_kernels(self,string,parameters_in):
        if string=='rbf':
            l = RBFKernel(ard_num_dims=self.Z.shape[1])
            ls=get_median_ls(self.Z).to(self.device)
            print(ls)
            l._set_lengthscale(ls)
            k = ScaleKernel(l)
            y_var = self.Y.var().item()
            k._set_outputscale(y_var)
            ls_obj=ls_init(k,self.Y_Z,self.Z,sigma=parameters_in['sigma'],its=parameters_in['init_its']).to(self.device)
            k = ls_obj.pre_train()
            print(l.lengthscale,k.outputscale)
            # l.lengthscale = torch.clip(l.lengthscale,min=2.0)
            print(l.lengthscale,k.outputscale)
            l.requires_grad_(False)
            k.requires_grad_(False)
        elif string == 'r_param_scaling':
            k = r_param_cholesky_scaling(k=self.k, Z=self.Z, X=self.X_hat,
                                         sigma=self.k.outputscale.item(),
                                         parametrize_Z=parameters_in['parametrize_Z']).to(self.device)
            print(self.k.outputscale,self.k.base_kernel.lengthscale)
            k.init_L()
        elif string == 'r_param_simple':
            k = r_param_cholesky_simpler(k=self.k, Z=self.Z, X=self.X_hat,
                                         parametrize_Z=parameters_in['parametrize_Z'], sigma=self.k.outputscale.item()).to(self.device)
            print(self.k.outputscale,self.k.base_kernel.lengthscale)
            k.init_L()

        return k

    def NLL_train(self,opt,mode='train',T=None):
        # torch.autograd.set_detect_anomaly(True)
        self.dataloader.dataset.set(mode)
        self.vi_obj.eval()
        obs_size=0.
        losses=[]
        for i, (X, x_cat, y) in enumerate(tqdm.tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            obs_size += y.shape[0]
            NLL = self.vi_obj.calc_NLL(y, X, T)
            losses.append(NLL)
        totloss = torch.stack(losses,dim=0).sum()
        tot_NLLs = totloss/obs_size
        opt.zero_grad()
        tot_NLLs.backward()
        opt.step()
        return tot_NLLs.item()
        # print('before',self.vi_obj.k.base_kernel.lengthscale)
        # def closure():
        #     obs_size=0.
        #     losses=0.
        #     print('inside',self.vi_obj.k.base_kernel.lengthscale)
        #     for i, (X, x_cat, y) in enumerate(tqdm.tqdm(self.dataloader)):
        #         X = X.to(self.device)
        #         y = y.to(self.device)
        #         obs_size += y.shape[0]
        #         with torch.no_grad():
        #             NLL = self.vi_obj.calc_NLL(y, X, T)
        #         losses += NLL.item()
        #     tot_NLLs = losses/obs_size
        #     print('inside',tot_NLLs)
        #     return torch.tensor([tot_NLLs]).float().to(self.device)
        # self.nll_opt.step(closure)
            # self.nll_opt.zero_grad()
            # NLL.backward()
            # self.nll_opt.step()


    def optimize_ls(self,opt,mode='val',T=None):
        best = np.inf
        pbar= tqdm.tqdm(range(500))
        for i in pbar:
            self.NLL_train(opt=opt,mode=mode,T=T)
            validation_loss_log_likelihood,r2,rmse=self.validation_loop(mode='val',T=T)
            pbar.set_description(f"NLL ls {validation_loss_log_likelihood}")
            if validation_loss_log_likelihood<best:
                best = validation_loss_log_likelihood
                self.dump_model(self.global_hyperit)


    def optimize_T(self,mode='train'):

        best = np.inf
        pbar = tqdm.tqdm(range(1000))
        t = torch.tensor([1.],requires_grad=True).float().to(self.device)
        T = torch.nn.Parameter(t)
        opt = torch.optim.Adam([T],1e-2)
        for i in pbar:
            b= T**2
            print('T',b.item())
            self.NLL_train(opt=opt, mode=mode, T=b)
            validation_loss_log_likelihood, r2, rmse = self.validation_loop(mode='val', T=b)
            pbar.set_description(f"NLL ls {validation_loss_log_likelihood}")
            if validation_loss_log_likelihood < best:
                best = validation_loss_log_likelihood
                b_best = b.item()
        return b_best
        # NLLs = []
        # RSMEs = []
        # R2s = []
        # t_list = np.linspace(1e-6,2,1000)
        # for t in t_list:
        #     validation_loss_log_likelihood,r2,rmse = self.validation_loop(mode,t)
        #     NLLs.append(validation_loss_log_likelihood)
        #     RSMEs.append(rmse)
        #     R2s.append(r2)
        # best_i = np.argmin(NLLs)
        # return t_list[best_i]

    def validation_loop(self,mode='val',T=None):
        self.vi_obj.eval()
        self.dataloader.dataset.set(mode)
        losses=0.0
        obs_size=0
        y_list = []
        y_pred_list = []
        for i,(X,x_cat,y) in enumerate(tqdm.tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            obs_size+=y.shape[0]
            with torch.no_grad():
                NLL = self.vi_obj.calc_NLL(y, X,T)
                y_pred = self.vi_obj.m_q(X)

            y_pred_list.append(y_pred)
            y_list.append(y)
            losses+=NLL.item()
        Y = torch.cat(y_list)
        Y_pred = torch.cat(y_pred_list)
        r2 = cuda_r2(Y_pred,Y)
        rmse = cuda_rmse(Y_pred,Y)
        validation_loss_log_likelihood = losses/obs_size
        print(f'held out {mode} nll: ',validation_loss_log_likelihood)
        print(f'held out {mode} R^2: ',r2)
        print(f'held out {mode} rmse: ',rmse)
        return validation_loss_log_likelihood,r2,rmse

    def train_loop(self,opt):
        self.vi_obj.train()
        self.dataloader.dataset.set('train')
        pbar= tqdm.tqdm(self.dataloader)
        for i,(X,x_cat,y) in enumerate(pbar):
            X=X.to(self.device)
            y=y.to(self.device)
            z_mask=torch.randperm(self.n)[:self.m]
            Z_prime = self.X[z_mask, :].to(self.device)
            log_loss,D=self.vi_obj.get_loss(y,X,Z_prime)
            # print(self.vi_obj.r.k.base_kernel.lengthscale)
            pbar.set_description(f"D: {D.item()} log_loss: {log_loss.item() }")
            tot_loss = D + log_loss
            opt.zero_grad()
            tot_loss.backward()
            opt.step()

    def train_loop_NLL(self,opt):
        self.vi_obj.train()
        self.dataloader.dataset.set('train')
        pbar= tqdm.tqdm(self.dataloader)
        for i,(X,x_cat,y) in enumerate(pbar):
            X=X.to(self.device)
            y=y.to(self.device)
            # z_mask=torch.randperm(self.n)[:self.m]
            # Z_prime = self.X[z_mask, :].to(self.device)
            # log_loss,D=self.vi_obj.get_loss(y,X,Z_prime)
            NLL = self.vi_obj.calc_NLL(y,X)/y.shape[0]
            # print(self.vi_obj.r.k.base_kernel.lengthscale)
            # pbar.set_description(f"D: {D.item()} log_loss: {log_loss.item() }")
            pbar.set_description(f"NLL lol: {NLL.item()}")
            tot_loss = NLL
            opt.zero_grad()
            tot_loss.backward()
            opt.step()

    def train_loop_m_q(self,opt):
        self.dataloader.dataset.set('train')
        self.vi_obj.train()
        loss = torch.nn.MSELoss()
        pbar= tqdm.tqdm(self.dataloader)
        for i,(X,x_cat,y) in enumerate(pbar):
            X=X.to(self.device)
            y=y.to(self.device)
            y_pred=self.vi_obj.mean_forward(X)
            tot_loss = loss(y_pred,y)
            pbar.set_description(f"MSE loss: {tot_loss.item()}")
            opt.zero_grad()
            tot_loss.backward()
            opt.step()
    def fit(self):
        best=-np.inf
        counter=0
        self.opt = torch.optim.Adam(self.vi_obj.parameters(), lr=self.lr)
        for i in range(self.train_params['epochs']):
            self.train_loop_m_q(self.opt)
            validation_loss,r2,rsme=self.validation_loop('val')
            if r2>best:
                best=r2
                self.dump_model(self.global_hyperit)
                counter=0
            else:
                counter+=1
                if counter>10:
                    break
        self.load_model(self.global_hyperit)
        self.opt = torch.optim.Adam(self.vi_obj.parameters(), lr=self.lr)
        self.vi_obj.r.k.requires_grad_(False)
        best=np.inf
        counter=0
        for i in range(self.train_params['epochs']):
            self.train_loop(self.opt)
            validation_loss,r2,rsme=self.validation_loop('val')
            if validation_loss<best:
                best=validation_loss
                self.dump_model(self.global_hyperit)
                counter=0
            else:
                counter+=1
                if counter>self.train_params['patience']:
                    break
        self.load_model(self.global_hyperit)

        # self.vi_obj.k.base_kernel.requires_grad_(True)
        # sampler = UniformSampler(minval=2, maxval=5.0, cuda=self.device)
        # sampler = GaussianSampler(mu=3, sigma=1.0, cuda=self.device)
        # self.nll_opt = SimulatedAnnealing(self.vi_obj.k.base_kernel.parameters(), sampler= sampler)
        T = self.optimize_T('train')
        self.vi_obj.zero_grad()
        self.vi_obj.k.base_kernel.requires_grad_(True)
        nll_opt = torch.optim.Adam(self.vi_obj.k.base_kernel.parameters(),lr=1e-2)
        self.optimize_ls(nll_opt,'train',T)
        self.load_model(self.global_hyperit)
        validation_loss,valr2,val_rsme=self.validation_loop('val',T)
        test_loss,testr2,test_rsme = self.validation_loop('test',T)
        self.create_NLL_plot('val',T)
        self.create_NLL_plot('test',T)

        return validation_loss,test_loss,valr2,testr2,val_rsme,test_rsme,T

    def create_NLL_plot(self,mode='val',T=None):
        self.vi_obj.eval()
        self.dataloader.dataset.set(mode)
        y_list = []
        y_pred_list = []
        vars = []
        for i, (X, x_cat, y) in enumerate(tqdm.tqdm(self.dataloader)):
            X = X.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                var_X = X.unsqueeze(1)
                var = self.vi_obj.r(var_X).squeeze() + self.sigma
                if T is not None:
                    var=var*T
                y_pred = self.vi_obj.m_q(X)
            y_pred_list.append(y_pred)
            y_list.append(y)
            vars.append(var)
        Y = torch.cat(y_list).cpu().numpy().squeeze()
        Y_pred = torch.cat(y_pred_list).cpu().numpy().squeeze()
        mse_per_ob = (Y-Y_pred)**2
        all_vars = torch.cat(vars).cpu().numpy()
        df = pd.DataFrame(np.stack([Y,Y_pred,mse_per_ob,all_vars],axis=1),columns=['Y','Y_pred','mse','var'])
        sns.lineplot(data=df,x='Y',y='Y')
        sns.scatterplot(data=df, x="Y", y="Y_pred")
        plt.savefig(self.save_path+f'{mode}_y_plots_{self.global_hyperit}.png')
        plt.clf()
        # sns.lineplot(data=df,x='mse',y='mse')
        sns.scatterplot(data=df, x="var", y="mse")
        plt.savefig(self.save_path + f'{mode}_mse_var_{self.global_hyperit}.png')
        plt.clf()

    def run(self):
        if os.path.exists(self.save_path + 'hyperopt_database.p'):
            return
        trials = Trials()
        best = fmin(fn=self,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.train_params['hyperits'],
                    trials=trials,
                    verbose=True)
        print(space_eval(self.hyperparameter_space, best))
        model_copy = dill.dumps(trials)
        pickle.dump(model_copy,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))


    def predict_mean(self,x_test):
        return self.vi_obj.mean_pred(x_test)

    def predict_uncertainty(self, x_test):
        return self.vi_obj.posterior_variance(x_test)

class experiment_classification_object():
    def __init__(self,hyper_param_space,VI_params,train_params,device='cuda:0'):
        self.train_params=train_params
        self.VI_params = VI_params
        self.device = device
        self.hyperopt_params = ['depth_x', 'width_x','depth_fc', 'bs', 'lr','m_P','sigma','transformation','m_factor']
        self.get_hyperparameterspace(hyper_param_space)
        self.generate_save_path()
        self.log_upper_bound = np.log(self.train_params['output_classes'])
        self.auc_interval = torch.from_numpy(np.linspace(0,self.log_upper_bound,100)).unsqueeze(0).to(device)
        self.global_hyperit=0

    def generate_save_path(self):
        model_name = self.train_params['model_name']
        savedir = self.train_params['savedir']
        dataset = self.train_params['dataset']
        fold = self.train_params['fold']
        seed = self.train_params['seed']
        self.save_path = f'{savedir}/{dataset}_seed={seed}_fold_idx={fold}_model={model_name}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self,parameters_in):
        self.dataloader_train,self.dataloader_val,self.dataloader_test=get_dataloaders(dataset_string=self.train_params['dataset'],
                                                                                       batch_size=parameters_in['bs'],
                                                                                       val_factor=self.train_params['val_factor']
                                                                                       )
        self.OOB_loader = get_dataloaders_OOB(
            dataset_string=self.train_params['dataset'],
            batch_size=parameters_in['bs'],
        )

        # self.dataloader = get_regression_dataloader(dataset=self.train_params['dataset'],fold=self.train_params['fold'],bs=self.train_params['bs'])
        x_list=[]
        y_list=[]
        for i,(X,y) in enumerate(tqdm.tqdm(self.dataloader_train)):
            x_list.append(X)
            y_list.append(y)
        self.X=torch.cat(x_list,dim=0)
        self.n= self.X.shape[0]
        self.m=int(round(self.X.shape[0]**0.5)*parameters_in['m_factor'])
        parameters_in['m'] = self.m
        parameters_in['init_its'] = self.train_params['init_its']
        self.Y=torch.cat(y_list,dim=0).unsqueeze(-1)

        if self.n>parameters_in['m']:
            z_mask=torch.randperm(self.n)[:parameters_in['m']]
            self.Z =  self.X[z_mask].flatten(1).float()
            self.X_hat =  self.X[torch.randperm(self.n)[:parameters_in['m']]].flatten(1).float()
            self.Y_Z= self.Y[z_mask,:].float()

        else:
            self.Z=self.X.flatten(1).float()
        self.k = self.get_kernels(self.VI_params['p_kernel'],parameters_in)
        self.r=torch.nn.ModuleList()
        for i in range(self.train_params['output_classes']):
            self.r.append(self.get_kernels(self.VI_params['q_kernel'],parameters_in))

        nn_params = {
            'cdim':self.train_params['cdim'],
            'output':self.train_params['output_classes'],
            'channels':[parameters_in['width_x']]*parameters_in['depth_x'],
            'image_size':self.train_params['image_size'],
            'transform': parameters_in['transformation'],
            'channels_fc':[parameters_in['width_x']]*parameters_in['depth_fc']
        }
        if self.train_params['m_q_choice']=='kernel_sum':
            nn_params['k'] = self.k
            nn_params['Z'] = self.Z
            self.m_q = conv_net_classifier_kernel(**nn_params).to(self.device)
        elif self.train_params['m_q_choice']=='CNN':
            self.m_q = conv_net_classifier(**nn_params).to(self.device)
        self.vi_obj=GVI_multi_classification(
                        Z = self.Z,
                        N=self.n,
                        m_q=self.m_q,
                        m_p=parameters_in['m_P'],
                        r_list=self.r,
                        k_list=self.k,
                        sigma=parameters_in['sigma'],
                        reg=self.VI_params['reg'],
                        APQ=self.VI_params['APQ']
                        ).to(self.device)
        self.lr = parameters_in['lr']
        val_acc,val_nll,val_ood_auc,val_ood_auc_prior,test_acc,test_nll,test_ood_auc,test_ood_auc_prior,T=self.fit()
        self.global_hyperit+=1
        return {'loss': -val_acc,
                'val_acc':val_acc,
                'val_nll':val_nll,
                'val_ood_auc':val_ood_auc,
                'val_ood_auc_prior':val_ood_auc_prior,
                'status': STATUS_OK,
                'test_acc': test_acc,
                'test_nll': test_nll,
                'test_ood_auc': test_ood_auc,
                'test_ood_auc_prior': test_ood_auc_prior,
                'net_params': nn_params,
                'T':T
                }

    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def get_kernels(self,string,params):
        if string=='rbf':
            l = RBFKernel(ard_num_dims=self.Z.shape[1])
            ls=get_median_ls(self.X).to(self.device)
            print(ls)
            l._set_lengthscale(ls)
            l.requires_grad_(True)
            k = ScaleKernel(l)
            k._set_outputscale(1.0)
            ls_obj=ls_init(k,self.Y_Z,self.Z,sigma=params['sigma'],its=params['init_its']).to(self.device)
            ls_obj.pre_train()
            print(l.lengthscale,k.outputscale)
            k.requires_grad_(False)
            l.requires_grad_(False)

        elif string == 'r_param_scaling':
            k = r_param_cholesky_scaling(k=self.k, Z=self.Z, X=self.X_hat,
                                         sigma=self.k.outputscale.item(),
                                         parametrize_Z=self.VI_params['parametrize_Z']).to(self.device)
            print(k.outputscale,k.base_kernel.lengthscale)
            k.init_L()

        return k

    def validation_loop(self,mode='val',T=None): #acc and nll
        self.vi_obj.eval()
        if mode=='val':
            dl=self.dataloader_val
        if mode=='test':
            dl=self.dataloader_test
        nll=0.0
        n=0
        acc=[]
        for i, (X, y) in enumerate(tqdm.tqdm(dl)):
            X = X.float().to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                NLL = self.vi_obj.calc_NLL(y, X,T)
                softmax_output=self.predict_mean(X)
                max_scores, max_idx_class = softmax_output.max(
                    dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
                acc.append((max_idx_class==y).squeeze())

            nll+=NLL.item()
            n+=y.shape[0]
        nll=nll/n
        acc=torch.cat(acc,dim=0).sum()/n
        print(f'held out {mode} acc: ',acc.item())
        print(f'held out {mode} nll: ',nll)
        return acc.item(),nll

    def get_auc(self,in_sample_preds,in_sample_labels,out_sample_preds,out_sample_labels,threshold):
        #Fucked up the calculation... redo
        in_sample_preds= in_sample_preds/self.log_upper_bound
        out_sample_preds=out_sample_preds/self.log_upper_bound
        preds=torch.cat([in_sample_preds,out_sample_preds],dim=0).cpu().numpy()
        truths=torch.cat([in_sample_labels,out_sample_labels],dim=0).cpu().numpy()
        AUC=roc_auc_score(truths,preds)
        # classification_in_sample = (in_sample_preds.unsqueeze(-1)<=threshold)==in_sample_labels.unsqueeze(-1) #horizontal axis are different cut-offs
        # y_axis = classification_in_sample.float().mean(0).cpu().numpy()
        # classification_out_sample = (out_sample_preds.unsqueeze(-1)<=threshold)==out_sample_labels.unsqueeze(-1) #horizontal axis are different cut-offs
        # x_axis = (1.- classification_out_sample.float().mean(0)).cpu().numpy()

        # AUC = torch.mean(classification.float()).item()
        return AUC

    def OOD_AUC(self,mode='val'):
        self.vi_obj.eval()
        if mode == 'val':
            dl = self.dataloader_val
        if mode == 'test':
            dl = self.dataloader_test

        entropies_in_sample=[]
        entropies_in_sample_prior=[]
        for i, (X, y) in enumerate(tqdm.tqdm(dl)):
            X = X.float().to(self.device)
            # y = y.to(self.device)
            with torch.no_grad():
                p=self.predict_mean(X)
                e = torch.sum(- p * torch.log(p),dim=1)
                e = torch.nan_to_num(e,nan=0.0)
                entropies_in_sample.append(e)

                p_prior=self.predict_mean_prior(X)
                e_prior = torch.sum(- p_prior * torch.log(p_prior),dim=1)
                e_prior = torch.nan_to_num(e_prior,nan=0.0)
                entropies_in_sample_prior.append(e_prior)

        entropies_in_sample = torch.cat(entropies_in_sample,dim=0)
        entropies_in_sample_prior = torch.cat(entropies_in_sample_prior,dim=0)
        true_labels_in_sample = torch.zeros_like(entropies_in_sample)

        entropies_out_sample=[]
        entropies_out_sample_prior=[]
        for i, (X, y) in enumerate(tqdm.tqdm(self.OOB_loader)):
            X = X.float().to(self.device)
            # y = y.to(self.device)
            with torch.no_grad():
                p=self.predict_mean(X)
                e = torch.sum(- p * torch.log(p),dim=1)
                e = torch.nan_to_num(e,nan=0.0)
                entropies_out_sample.append(e)

                p_prior=self.predict_mean_prior(X)
                e_prior = torch.sum(- p_prior * torch.log(p_prior),dim=1)
                e_prior = torch.nan_to_num(e_prior,nan=0.0)
                entropies_out_sample_prior.append(e_prior)
        entropies_out_sample = torch.cat(entropies_out_sample, dim=0)
        entropies_out_sample_prior = torch.cat(entropies_out_sample_prior, dim=0)
        true_labels_out_sample=torch.ones_like(entropies_out_sample)
        # in_sample_preds, in_sample_labels, out_sample_preds, out_sample_labels, threshold
        auc=self.get_auc(entropies_in_sample,true_labels_in_sample,entropies_out_sample,true_labels_out_sample,self.auc_interval)
        auc_prior=self.get_auc(entropies_in_sample_prior,true_labels_in_sample,entropies_out_sample_prior,true_labels_out_sample,self.auc_interval)
        print('standard AUC OOD: ', auc)
        print('priornAUC OOD: ', auc_prior)

        return auc,auc_prior

    def train_loop(self,opt):
        self.vi_obj.train()
        pbar= tqdm.tqdm(self.dataloader_train)
        for i,(X,y) in enumerate(pbar):
            # with autograd.detect_anomaly():
            X=X.float().to(self.device)
            y=y.to(self.device)
            z_mask=torch.randperm(self.n)[:self.m]
            Z_prime = self.X[z_mask, :].to(self.device)
            log_loss,D=self.vi_obj.get_loss(y,X,Z_prime)
            # log_loss,D=self.vi_obj.get_loss(y,X)
            pbar.set_description(f"D: {D.item()} log_loss: {log_loss.item() }")
            tot_loss = D + log_loss
            opt.zero_grad()
            tot_loss.backward()
            opt.step()

    def train_loop_acc(self,opt):
        self.vi_obj.train()
        pbar= tqdm.tqdm(self.dataloader_train)
        loss = nn.CrossEntropyLoss()
        for i,(X,y) in enumerate(pbar):
            X=X.float().to(self.device)
            y=y.to(self.device)
            y_hat=self.vi_obj.mean_forward(X)
            # log_loss,D=self.vi_obj.get_loss(y,X)
            tot_loss = loss(y_hat,y)

            pbar.set_description(f"CE loss: {tot_loss.item()}")
            opt.zero_grad()
            tot_loss.backward()
            opt.step()
    def optimize_T(self,mode='val'):
        NLLs = []
        t_list = np.linspace(0.1,5.0,100)
        for t in t_list:
            _,validation_loss_log_likelihood = self.validation_loop(mode,t)
            NLLs.append(validation_loss_log_likelihood)
        best_i = np.argmin(NLLs)
        return t_list[best_i]

    def fit(self):
        best=0.0
        counter=0

        self.opt=torch.optim.Adam(self.vi_obj.parameters(),lr=self.lr)
        for i in range(self.train_params['epochs']):
            self.train_loop_acc(self.opt)
            val_acc, val_nll = self.validation_loop('val')
            # print(self.k.outputscale, self.k.base_kernel.lengthscale)
            if val_acc > best:
                best = val_acc
                print('woho new best model!')
                self.dump_model(self.global_hyperit)
                counter = 0
            else:
                counter += 1
                if counter > self.train_params['patience']:
                    break
        best=np.inf
        counter=0
        self.load_model(self.global_hyperit)
        self.opt=torch.optim.Adam(self.vi_obj.parameters(),lr=self.lr)
        for i in range(self.train_params['epochs']):
            self.train_loop(self.opt)
            val_acc,val_nll=self.validation_loop('val')
            if val_nll<best:
                best=val_nll
                print('woho new best model!')
                self.dump_model(self.global_hyperit)
                counter = 0
            else:
                counter+=1
                if counter>self.train_params['patience']:
                    break

        self.load_model(self.global_hyperit)
        T = self.optimize_T(mode='val')
        test_ood_auc,test_ood_auc_prior = self.OOD_AUC('test')
        val_ood_auc,val_ood_auc_prior = self.OOD_AUC('val')
        val_acc,val_nll=self.validation_loop('val',T)
        test_acc,test_nll=self.validation_loop('test',T)
        return val_acc,val_nll,val_ood_auc,val_ood_auc_prior,test_acc,test_nll,test_ood_auc,test_ood_auc_prior,T
        # except Exception as e:
        #     torch.cuda.empty_cache()
        #     return -99999,-99999,-99999,-99999,-99999,-99999

    def dump_model(self,hyperit):
        model_copy = dill.dumps(self.vi_obj)
        torch.save(model_copy, self.save_path+f'best_model_{hyperit}.pt')

    def load_model(self,hyperit):
        model_copy=torch.load(self.save_path+f'best_model_{hyperit}.pt')
        self.vi_obj=dill.loads(model_copy)
    def run(self):
        if os.path.exists(self.save_path + 'hyperopt_database.p'):
            return
        trials = Trials()
        best = fmin(fn=self,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.train_params['hyperits'],
                    trials=trials,
                    verbose=True)
        print(space_eval(self.hyperparameter_space, best))
        model_copy = dill.dumps(trials)
        pickle.dump(model_copy,
                    open(self.save_path + 'hyperopt_database.p',
                         "wb"))

    def predict_mean(self,x_test):
        return self.vi_obj.mean_pred(x_test)

    def predict_mean_prior(self, x_test):
        return self.vi_obj.mean_pred_prior(x_test)

    def predict_uncertainty(self, x_test):
        return self.vi_obj.posterior_variance(x_test)





