import torch
from gwi_VI.mean_models import *
from gwi_VI.models import *
from gpytorch.kernels import RBFKernel,LinearKernel,CosineKernel
from utils.dataloaders import *
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


class experiment_object():
    def __init__(self,X,Y,nn_params,VI_params,train_params):
        self.X = X
        self.Y = Y
        self.device=train_params['device']

        self.VI_params = VI_params
        self.train_params = train_params
        self.r=self.get_kernels(self.VI_params['q_kernel'],True)
        self.k=self.get_kernels(self.VI_params['p_kernel'],False)
        nn_params['d_in_x']=self.X.shape[1]
        self.m_q = feature_map(**nn_params).to(self.device)
        self.m_p = self.VI_params['m_p']
        self.vi_obj=GWI(m_q=self.m_q,
                        m_p=self.m_p,
                        r=self.r,
                        k=self.k,
                        X=self.X,
                        sigma=self.VI_params['sigma'],
                        m=self.VI_params['m'],
                        reg=self.VI_params['reg']
                        ).to(self.device)
        self.vi_obj.calculate_U()
        dataset=general_custom_dataset(X,Y,nn_params['cat_size_list'])
        self.dataloader=custom_dataloader(dataset,train_params['bs'])

    def get_kernels(self,string,is_q=False):
        if string=='rbf':
            k=RBFKernel()
            ls=get_median_ls(self.X)
            k._set_lengthscale(ls)
        if is_q:
            k.requires_grad_(True)
        else:
            k.requires_grad_(False)
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
            if tr_m:
                ll,reg=self.vi_obj.likelihood_reg(y,X)
                loss = ll+reg
            else:
                hard_trace,easy_trace=self.vi_obj.calc_hard_tr_term()
                loss = hard_trace+easy_trace
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(loss.item())

    def fit(self):
        # for p in self.vi_obj.parameters():
        #     print(p)
        opt=torch.optim.Adam(self.vi_obj.parameters(),lr=self.train_params['lr'])
        for i in range(self.train_params['epochs']):
            self.train_loop(opt,True)
        for i in range(self.train_params['epochs']):
            self.train_loop(opt,False)

        self.vi_obj.invert_r_mat()

    def predict_mean(self,x_test):
        return self.vi_obj.mean_pred(x_test)

    def predict_uncertainty(self, x_test):
        return self.vi_obj.posterior_variance(x_test)


