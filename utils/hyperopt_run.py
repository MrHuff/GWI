import torch
from gwi_VI.mean_models import *
from gwi_VI.models import *
from gpytorch.kernels import RBFKernel,LinearKernel,CosineKernel,ScaleKernel
from utils.dataloaders import *
from utils.regression_dataloaders import *
import tqdm


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

def get_median_ls( X, Y=None):
    with torch.no_grad():
        if Y is None:
            d = covar_dist(x1=X, x2=X)
        else:
            d = covar_dist(x1=X, x2=Y)
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
        self.vi_obj=GWI(m_q=self.m_q,
                        m_p=self.m_p,
                        r=self.r,
                        k=self.k,
                        X=self.X,
                        sigma=self.VI_params['sigma'],
                        Z=self.Z,
                        reg=self.VI_params['reg']
                        ).to(self.device)
        self.vi_obj.calculate_U()
        dataset=general_custom_dataset(X,Y,nn_params['cat_size_list'])
        self.dataloader=custom_dataloader(dataset,train_params['bs'])

    def get_kernels(self,string,is_q=False):
        if string=='rbf':
            l = RBFKernel()
            ls=get_median_ls(self.X).to(self.device)
            l._set_lengthscale(ls)
            k = ScaleKernel(l)
            k._set_outputscale(self.VI_params['y_var'])
            ls_obj=ls_init(k,self.Y_Z,self.Z,sigma=self.VI_params['sigma']).to(self.device)
            ls_obj.pre_train()
            print(l.lengthscale,k.outputscale)

            k.requires_grad_(False)
        elif string=='r_param':
            # l = RBFKernel()
            # ls=get_median_ls(self.X)
            # l._set_lengthscale(ls)
            # p = ScaleKernel(l)
            # p._set_outputscale(self.VI_params['y_var'])
            # k = r_param(k=self.k,Z=self.Z,d=self.VI_params['r'])
            k = r_param_cholesky(k=self.k,Z=self.Z,X=self.X_hat,sigma=self.VI_params['sigma']).to(self.device)
            k.init_L()
            # k.requires_grad_(True)
        # if is_q:
        #
        # else:
        return k

    def validation_loop(self,mode='val'):
        pass

    def train_loop(self,opt,tr_m=True):
        self.dataloader.dataset.set('train')
        for i,(X,x_cat,y) in tqdm.tqdm(enumerate(self.dataloader)):
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
        # for p in self.vi_obj.parameters():
        #     print(p)
        opt=torch.optim.Adam(self.vi_obj.parameters(),lr=self.train_params['lr'])
        for i in range(self.train_params['epochs']):
            self.train_loop(opt,True)

        print(self.vi_obj.r.scale)
        # for i in range(self.train_params['epochs']):
        #     self.train_loop(opt,False)


    def predict_mean(self,x_test):
        return self.vi_obj.mean_pred(x_test)

    def predict_uncertainty(self, x_test):
        return self.vi_obj.posterior_variance(x_test)


class mvp_experiment_regression_object():
    def __init__(self,nn_params,VI_params,train_params):
        self.dataset = train_params['dataset']
        self.fold = train_params['fold']

        self.dataloader = get_regression_dataloader(dataset=self.dataset,fold=self.fold,bs=train_params['bs'])
        self.n,self.d = self.dataloader.dataset.train_X.shape
        self.X=self.dataloader.dataset.train_X
        self.Y=self.dataloader.dataset.train_y


        self.device=train_params['device']

        self.VI_params = VI_params
        self.train_params = train_params

        nn_params['d_in_x']=self.d
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
        self.vi_obj=GWI(m_q=self.m_q,
                        m_p=self.m_p,
                        r=self.r,
                        k=self.k,
                        X=self.X,
                        sigma=self.VI_params['sigma'],
                        Z=self.Z,
                        reg=self.VI_params['reg']
                        ).to(self.device)
        self.vi_obj.calculate_U()


    def get_kernels(self,string,is_q=False):
        if string=='rbf':
            l = RBFKernel()
            ls=get_median_ls(self.X).to(self.device)
            l._set_lengthscale(ls)
            k = ScaleKernel(l)
            k._set_outputscale(self.VI_params['y_var'])
            ls_obj=ls_init(k,self.Y_Z,self.Z,sigma=self.VI_params['sigma']).to(self.device)
            ls_obj.pre_train()
            print(l.lengthscale,k.outputscale)

            k.requires_grad_(False)
        elif string=='r_param':
            # l = RBFKernel()
            # ls=get_median_ls(self.X)
            # l._set_lengthscale(ls)
            # p = ScaleKernel(l)
            # p._set_outputscale(self.VI_params['y_var'])
            # k = r_param(k=self.k,Z=self.Z,d=self.VI_params['r'])
            k = r_param_cholesky(k=self.k,Z=self.Z,X=self.X_hat,sigma=self.VI_params['sigma']).to(self.device)
            k.init_L()
            # k.requires_grad_(True)
        # if is_q:
        #
        # else:
        return k

    def validation_loop(self,mode='val'):
        pass

    def train_loop(self,opt,tr_m=True):
        self.dataloader.dataset.set('train')
        for i,(X,x_cat,y) in tqdm.tqdm(enumerate(self.dataloader)):
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
        # for p in self.vi_obj.parameters():
        #     print(p)
        opt=torch.optim.Adam(self.vi_obj.parameters(),lr=self.train_params['lr'])
        for i in range(self.train_params['epochs']):
            self.train_loop(opt,True)

        print(self.vi_obj.r.scale)
        # for i in range(self.train_params['epochs']):
        #     self.train_loop(opt,False)


    def predict_mean(self,x_test):
        return self.vi_obj.mean_pred(x_test)

    def predict_uncertainty(self, x_test):
        return self.vi_obj.posterior_variance(x_test)


