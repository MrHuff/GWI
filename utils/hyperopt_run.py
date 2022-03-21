import numpy as np
import torch
from gwi_VI.mean_models import *
from gwi_VI.models import *
from gpytorch.kernels import RBFKernel,LinearKernel,CosineKernel,ScaleKernel
from utils.dataloaders import *
from utils.regression_dataloaders import *
import tqdm
from hyperopt import hp,tpe,Trials,fmin,space_eval,STATUS_OK,STATUS_FAIL,rand
import pickle
from utils.classification_dataloaders import *
import dill
import torch.autograd as autograd

def cuda_r2(y_pred,y):
    res = (y_pred-y)**2
    return (1. - res.mean()/y.var()).item()

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
                        k=self.k,
                        sigma=self.VI_params['sigma'],
                        reg=self.VI_params['reg'],
                        APQ=self.VI_params['APQ']
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
            k = r_param_cholesky_scaling(k=self.k, Z=self.Z, X=self.X_hat, sigma=self.k.outputscale.item()).to(
                self.device)
            k.init_L()
            # k.requires_grad_(True)
        # if is_q:
        #
        # else:
        return k

    def train_loop(self,opt,tr_m=True):
        self.dataloader.dataset.set('train')
        for i,(X,x_cat,y) in enumerate(tqdm.tqdm(self.dataloader)):
            if not isinstance(x_cat,list):
                x_cat=x_cat.to(self.device)
            X=X.to(self.device)
            y=y.to(self.device)
            log_loss,D=self.vi_obj.get_loss(y,X)
            print('D: ',D.item())
            print('log_loss: ',log_loss.item())
                # print(self.r.lengthscale)
            tot_loss = D + log_loss
            opt.zero_grad()
            tot_loss.backward()
            opt.step()

    def fit(self):
        opt=torch.optim.Adam(self.vi_obj.parameters(),lr=self.train_params['lr'])
        for i in range(self.train_params['epochs']):
            self.train_loop(opt,True)

        print(self.vi_obj.r.scale)

    def predict_mean(self,x_test):
        return self.vi_obj.mean_pred(x_test)

    def predict_uncertainty(self, x_test):
        return self.vi_obj.posterior_variance(x_test)

class experiment_regression_object():
    def __init__(self,hyper_param_space,VI_params,train_params,device='cuda:0'):
        self.train_params=train_params
        self.VI_params = VI_params
        self.device = device
        self.hyperopt_params = ['transformation', 'depth_x', 'width_x', 'bs', 'lr','m_P','sigma','m']
        self.get_hyperparameterspace(hyper_param_space)
        self.generate_save_path()
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
        self.dataloader = get_regression_dataloader(dataset=self.train_params['dataset'],fold=self.train_params['fold'],bs=parameters_in['bs'])
        self.n,self.d = self.dataloader.dataset.train_X.shape
        self.X=self.dataloader.dataset.train_X
        self.Y=self.dataloader.dataset.train_y
        mean_y_train = self.Y.mean().item()
        nn_params = {
            'd_in_x' : self.d,
            'cat_size_list': [],
            'output_dim' : 1,
            'transformation':parameters_in['transformation'],
            'layers_x': [parameters_in['width_x']]*parameters_in['depth_x'],
        }
        if self.train_params['m_q_choice']=='kernel_sum':
            nn_params['k'] = self.k
            nn_params['Z'] = self.Z
            self.m_q = kernel_feature_map_regression(**nn_params).to(self.device)
        elif self.train_params['m_q_choice']=='mlp':
            self.m_q = feature_map(**nn_params).to(self.device)

        if self.n>parameters_in['m']:
            z_mask=torch.randperm(self.n)[:parameters_in['m']]
            self.Z =  self.X[z_mask, :]
            self.Y_Z= self.Y[z_mask,:]
            self.X_hat =  self.X[torch.randperm(self.n)[:parameters_in['m']], :]
        else:
            self.Z=self.X
        self.k=self.get_kernels(self.VI_params['p_kernel'],parameters_in)
        self.r=self.get_kernels(self.VI_params['q_kernel'],parameters_in)
        self.vi_obj=GWI(
                        N=self.X.shape[0],
                        m_q=self.m_q,
                        m_p=mean_y_train*parameters_in['m_P'],
                        r=self.r,
                        k=self.k,
                        sigma=parameters_in['sigma'],
                        reg=self.VI_params['reg'],
                        APQ=self.VI_params['APQ']
                        ).to(self.device)
        self.opt=torch.optim.Adam(self.vi_obj.parameters(),lr=parameters_in['lr'])

        val_loss,test_loss=self.fit()
        self.global_hyperit+=1
        return  {
                'loss': val_loss,
                'status': STATUS_OK,
                'test_loss': test_loss,
                 'net_params':nn_params
                }


    def get_hyperparameterspace(self,hyper_param_space):
        self.hyperparameter_space = {}
        for string in self.hyperopt_params:
            self.hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])

    def get_kernels(self,string,parameters_in):
        if string=='rbf':
            l = RBFKernel(ard_num_dims=self.Z.shape[1])
            ls=get_median_ls(self.X).to(self.device)
            print(ls)
            l._set_lengthscale(ls)
            k = ScaleKernel(l)
            y_var = self.Y.var().item()
            k._set_outputscale(y_var)
            ls_obj=ls_init(k,self.Y_Z,self.Z,sigma=parameters_in['sigma']).to(self.device)
            k = ls_obj.pre_train()
            print(l.lengthscale,k.outputscale)
            k.requires_grad_(False)
        elif string == 'r_param_scaling':
            k = r_param_cholesky_scaling(k=self.k, Z=self.Z, X=self.X_hat,
                                         sigma=self.k.outputscale.item(),
                                         parametrize_Z=self.VI_params['parametrize_Z']).to(
                self.device)
            k.init_L()

        return k

    def validation_loop(self,mode='val'):
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
                NLL = self.vi_obj.calc_NLL(y, X)
                y_pred = self.vi_obj.m_q(X)
            y_pred_list.append(y_pred)
            y_list.append(y)
            losses+=NLL.item()
        Y = torch.cat(y_list)
        Y_pred = torch.cat(y_pred_list)
        r2 = cuda_r2(Y_pred,Y)
        validation_loss_log_likelihood = losses/obs_size
        print(f'held out {mode} nll: ',validation_loss_log_likelihood)
        print(f'held out {mode} R^2: ',r2)
        return validation_loss_log_likelihood

    def train_loop(self,opt):
        self.vi_obj.train()
        self.dataloader.dataset.set('train')
        pbar= tqdm.tqdm(self.dataloader)

        for i,(X,x_cat,y) in enumerate(pbar):
            X=X.to(self.device)
            y=y.to(self.device)

            with autograd.detect_anomaly():
                log_loss,D=self.vi_obj.get_loss(y,X)
                pbar.set_description(f"D: {D.item()} log_loss: {log_loss.item() }")
                tot_loss = D + log_loss
                opt.zero_grad()
                tot_loss.backward()
                opt.step()

    def fit(self):
        best=np.inf
        counter=0
        for i in range(self.train_params['epochs']):
            self.train_loop(self.opt)
            validation_loss=self.validation_loop('val')
            if validation_loss<best:
                best=validation_loss
                self.dump_model(self.global_hyperit)
            else:
                counter+=1
                if counter>self.train_params['patience']:
                    break
        self.load_model(self.global_hyperit)
        validation_loss=self.validation_loop('val')
        test_loss = self.validation_loop('test')
        return validation_loss,test_loss
    def clean_L_grads(self):
        self.vi_obj.r.L.grad=torch.nan_to_num(self.vi_obj.r.L.grad,nan=-1e-2)


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
        self.hyperopt_params = ['depth_x', 'width_x','depth_fc', 'bs', 'lr','m_P','sigma','transformation','m']
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
                        N=self.n,
                        m_q=self.m_q,
                        m_p=parameters_in['m_P'],
                        r_list=self.r,
                        k_list=self.k,
                        sigma=parameters_in['sigma'],
                        reg=self.VI_params['reg'],
                        APQ=self.VI_params['APQ']
                        ).to(self.device)
        self.opt=torch.optim.Adam(self.vi_obj.parameters(),lr=parameters_in['lr'])
        val_acc,val_nll,val_ood_auc,test_acc,test_nll,test_ood_auc=self.fit()
        self.global_hyperit+=1
        return {'loss': val_acc,
                'val_acc':val_acc,
                'val_nll':val_nll,
                'val_ood_auc':val_ood_auc,
                'status': STATUS_OK,
                'test_acc': test_acc,
                'test_nll': test_nll,
                'test_ood_auc': test_ood_auc,
                'net_params': nn_params
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
            ls_obj=ls_init(k,self.Y_Z,self.Z,sigma=params['sigma']).to(self.device)
            ls_obj.pre_train()
            print(l.lengthscale,k.outputscale)
            k.requires_grad_(False)
            l.requires_grad_(False)

        elif string == 'r_param_scaling':
            k = r_param_cholesky_scaling(k=self.k, Z=self.Z, X=self.X_hat,
                                         sigma=self.k.outputscale.item(),
                                         parametrize_Z=self.VI_params['parametrize_Z']).to(self.device)
            k.init_L()

        return k

    def validation_loop(self,mode='val'): #acc and nll
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
                NLL = self.vi_obj.calc_NLL(y, X)
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

    def get_auc(self,entropy,true_labels,threshold):
        classification = (entropy.unsqueeze(-1)<=threshold)==true_labels.unsqueeze(-1)
        AUC = torch.mean(classification.float()).item()
        print('AUC OOD: ', AUC)
        return AUC

    def OOD_AUC(self,mode='val'):
        self.vi_obj.eval()
        if mode == 'val':
            dl = self.dataloader_val
        if mode == 'test':
            dl = self.dataloader_test

        true_labels=[]
        entropies_in_sample=[]
        for i, (X, y) in enumerate(tqdm.tqdm(dl)):
            X = X.float().to(self.device)
            # y = y.to(self.device)
            with torch.no_grad():
                p=self.predict_mean(X)
                e = torch.sum(- p * torch.log(p),dim=1)
                entropies_in_sample.append(e)
        entropies_in_sample = torch.cat(entropies_in_sample,dim=0)
        true_labels.append(torch.ones_like(entropies_in_sample))

        entropies_out_sample=[]
        for i, (X, y) in enumerate(tqdm.tqdm(self.OOB_loader)):
            X = X.float().to(self.device)
            # y = y.to(self.device)
            with torch.no_grad():
                p=self.predict_mean(X)
                e = torch.sum(- p * torch.log(p),dim=1)
                entropies_out_sample.append(e)
        entropies_out_sample = torch.cat(entropies_out_sample, dim=0)
        true_labels.append(torch.zeros_like(entropies_out_sample))
        all_entropies = torch.cat([entropies_in_sample,entropies_out_sample],dim=0)
        true_labels = torch.cat(true_labels,dim=0)
        auc=self.get_auc(all_entropies,true_labels,self.auc_interval)
        return auc

    def train_loop(self,opt):
        self.vi_obj.train()
        pbar= tqdm.tqdm(self.dataloader_train)
        for i,(X,y) in enumerate(pbar):
            with autograd.detect_anomaly():

                X=X.float().to(self.device)
                y=y.to(self.device)

                log_loss,D=self.vi_obj.get_loss(y,X)
                pbar.set_description(f"D: {D.item()} log_loss: {log_loss.item() }")

                tot_loss = D + log_loss
                opt.zero_grad()
                tot_loss.backward()
                opt.step()

    def clean_L_grads(self):
        for r in self.vi_obj.r:
            r.L.grad=torch.nan_to_num(r.L.grad,nan=0.0)

    def fit(self):
        best=0.0
        counter=0
        # try:
        for i in range(self.train_params['epochs']):
            self.train_loop(self.opt)
            val_acc,val_nll=self.validation_loop('val')
            print(self.k.outputscale,self.k.base_kernel.lengthscale)
            if val_acc>best:
                best=val_acc
                print('woho new best model!')
                self.dump_model(self.global_hyperit)
            else:
                counter+=1
                if counter>self.train_params['patience']:
                    break
        test_ood_auc = self.OOD_AUC('test')
        val_ood_auc = self.OOD_AUC('val')
        val_acc,val_nll=self.validation_loop('val')
        test_acc,test_nll=self.validation_loop('test')
        return val_acc,val_nll,val_ood_auc,test_acc,test_nll,test_ood_auc
        # except Exception as e:
        #     torch.cuda.empty_cache()
        #     return -99999,-99999,-99999,-99999,-99999,-99999

    def dump_model(self,hyperit):
        model_copy = dill.dumps(self.vi_obj)
        torch.save(model_copy, self.save_path+f'best_model_{hyperit}.pt')

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





