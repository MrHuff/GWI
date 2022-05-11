import torch
import tqdm
from gpytorch.kernels import RBFKernel
import gpytorch
from scipy.special import roots_hermite
import numpy as np
from torch.distributions.normal import Normal
import gc
from gwi_VI.mean_models import feature_map
CDF_APPROX_COEFF=1.65451
sqrt_pi=np.pi**0.5
log2pi=np.log(np.pi*2)

def ensure_pos_diag(L):
    v=torch.diag(L)
    v = torch.clamp(v,min=1e-6)
    mask = torch.diag(torch.ones_like(v))
    L = mask * torch.diag(v) + (1. - mask) * L
    return L

def ensure_pos_diag_svgp(K,cap=1e-1):
    v=torch.diag(K)
    v[v<=0]=cap
    mask = torch.diag(torch.ones_like(v))
    K = mask * torch.diag(v) + (1. - mask) * K
    return K

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class init_kernel_GP(torch.nn.Module):

    def calc_nll(self,y,val_output,noise):
        pred = val_output.mean
        rXX = val_output.variance + noise.item()
        vec=(y-pred)**2
        a = (0.5 * torch.log(rXX)).sum()
        b = (0.5 * vec / rXX).sum()
        return ((a + b) + self.const * pred.shape[0])/pred.shape[0]

    def __init__(self,y,Z,empirical_sigma,dataloader,its=25):
        super(init_kernel_GP, self).__init__()
        self.empirical_sigma = empirical_sigma
        self.its = its
        self.dataloader = dataloader
        self.register_buffer('val_X',self.dataloader.dataset.val_X.clone())
        self.register_buffer('val_Y',self.dataloader.dataset.val_y.clone())
        self.register_buffer('test_X',self.dataloader.dataset.test_X.clone())
        self.register_buffer('test_Y',self.dataloader.dataset.test_y.clone())
        self.log_empirical_sigma = np.log(self.empirical_sigma)
        self.const = (0.5 * log2pi + self.log_empirical_sigma).item()
        self.its = its
        self.register_buffer('X',Z)
        self.register_buffer('Y',y)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self.X, self.Y.squeeze(), self.likelihood)

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in tqdm.tqdm(range(self.its)):
            self.model.train()
            self.likelihood.train()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from self.vi_obj
            output = self.model(self.X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.Y.squeeze())
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.model.eval()
                self.likelihood.eval()
                val_output = self.model(self.val_X)
                val_NLL = self.calc_nll(self.val_Y.squeeze(),val_output,self.model.likelihood.noise)
                print(f'val NLL: {val_NLL.item()}')
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_output = self.model(self.test_X)
            test_NLL = self.calc_nll(self.test_Y.squeeze(), test_output,self.model.likelihood.noise)
        print(f'test NLL: {test_NLL.item()}')

        return self.model.covar_module,self.model.likelihood.noise.item()

class ls_init(torch.nn.Module):
    def __init__(self,k,y,Z,sigma,its=25):
        super(ls_init, self).__init__()
        self.its = its
        self.register_buffer('eye',torch.eye(Z.shape[0]))
        self.register_buffer('Z',Z)
        self.register_buffer('Y',y)

        self.k=k

    def objective(self,L):
        inv=torch.linalg.solve_triangular(L,self.Y)
        return -torch.diag(L).log().mean()-0.5*torch.mean(inv**2)

    def pre_train(self):
        opt = torch.optim.Adam(self.k.parameters(),lr=1e-1)
        for i in tqdm.tqdm(range(self.its)):
            opt.zero_grad()
            L=torch.linalg.cholesky(self.k(self.Z).evaluate()+self.eye*1e-2)
            loss= -self.objective(L)
            print(loss)
            loss.backward()
            opt.step()
        return self.k

class KRR_mean(torch.nn.Module):
    def __init__(self,r):
        super(KRR_mean, self).__init__()
        self.r = r
        self.alpha = torch.nn.Parameter(torch.randn(self.r.Z.shape[0],1))
    def forward(self,X,x_cat=[]):
        return self.r.k(X,self.r.Z).evaluate()@self.alpha

class neural_network_kernel_lol(torch.nn.Module):
    def __init__(self, k,Z,sigma,m_Q,nn_params,m):
        super(neural_network_kernel_lol, self).__init__()
        self.m = m
        self.C = 1.0
        self.sigma_f = sigma
        self.phi = m_Q
        self.k = k
        self.register_buffer('Z',Z)
        # self.final_layer = torch.nn.Linear(nn_params['layers_x'][-1],self.m)

    def phi_forward(self,x):
        # with torch.no_grad():
        layer = self.phi.covariate_net((x,[]))
        # layer = self.final_layer(layer)
        # layer = self.phi.covariate_net.up_to_layer((x, []),1)
        # return torch.sigmoid(layer)
        return layer
    def forward(self,x1,x2=None):

        if x2 is None:
            if len(x1.shape) == 3:
                v =self.phi_forward(x1.squeeze(1))
                return torch.sum(v**2,dim=1,keepdim=True).unsqueeze(-1)*self.sigma_f#
            else:
                v = self.phi_forward(x1)
                return v@v.t() * self.sigma_f#/self.m
        else:
            if len(x1.shape) == 3:
                v_1 = self.phi_forward(x1.squeeze(1))
                v_2 = self.phi_forward(x2.squeeze(1))
                return torch.sum(v_1 * v_2,dim=1,keepdim=True).unsqueeze(-1) * self.sigma_f
            else:
                v_1 = self.phi_forward(x1)
                v_2 = self.phi_forward(x2)
                return v_1 @ v_2.t() * self.sigma_f #/self.m


class r_param_cholesky_scaling(torch.nn.Module):
    def __init__(self, k, Z, X, sigma, scale_init=1.0, parametrize_Z=False):
        super(r_param_cholesky_scaling, self).__init__()
        self.k = k
        self.scale = torch.nn.Parameter(torch.ones(1)*scale_init)
        self.register_buffer('eye',torch.eye(Z.shape[0]))
        self.parametrize_Z = parametrize_Z
        if parametrize_Z:
            self.Z = torch.nn.Parameter(Z)
        else:
            self.register_buffer('Z',Z)
        self.register_buffer('X',X)
        self.sigma=sigma
        self.cap = 1e-1
    def init_L(self):
        with torch.no_grad():
            kx = self.k(self.Z, self.X).evaluate()
            self.kzz = self.k(self.Z).evaluate()
        L_set = False
        for reg in [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:#,1e-1,1.0,5.0,10.]:
            try:
                with torch.no_grad():
                    chol_L = torch.linalg.cholesky(self.kzz + kx @ kx.t() / self.sigma + self.eye*reg)
                    L = torch.linalg.cholesky(torch.cholesky_inverse(chol_L))
                    # L = torch.linalg.cholesky(torch.inverse(self.kzz  + self.eye*self.sigma))
                    self.L = torch.nn.Parameter(L)
                    L_set = True
            except Exception as e:
                gc.collect()
                print('cholesky init error: ',reg)
                print(e)
                    # torch.cuda.empty_cache()
        if not L_set:
            print("Welp this is awkward, it don't want to factorize")

            gc.collect()
            # torch.cuda.empty_cache()
            L = torch.randn_like(self.kzz) * 0.1
            self.L = torch.nn.Parameter(L)
        inv_set=False
        for reg in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:#, 1e-1, 1.0,2.5,5.0,10.]:
            self.reg=reg
            try:
                if not self.parametrize_Z:
                    with torch.no_grad():
                        self.register_buffer('inv_L', torch.linalg.cholesky(self.kzz + self.eye * self.reg))
                        self.register_buffer('inv_Z',torch.cholesky_inverse(self.inv_L))
                    inv_set=True
            except Exception as e:
                gc.collect()
                print(e)
                print('cholesky inv error: ',reg)
                pass
        if not inv_set:
            gc.collect()
            self.parametrize_Z=True
    def forward(self,x1,x2=None):
        L = torch.tril(self.L)+self.eye*self.reg
        L = ensure_pos_diag(L)
        Z= self.Z
        if x2 is None:
            kzx = self.k(Z, x1).evaluate()
            t= L.t() @ kzx #L\cdot k(Z,X)

            if self.parametrize_Z:
                kzz = self.k(Z).evaluate()
                chol_z = torch.linalg.cholesky(kzz+self.eye*self.reg)
                sol = torch.cholesky_solve(kzx,chol_z)
            else:
                # sol = torch.cholesky_solve(kzx, self.inv_L)
                sol = self.inv_Z @ kzx
            if len(t.shape)==3: #t=[L^T k(Z,X_i),L^T k(Z,X_{i+1}),]
                T_mat = t.permute(0,2,1) @ t #T_mat ok
                inverse_part = kzx.permute(0,2,1)@sol
                out = self.k(x1).evaluate()- inverse_part + T_mat#/self.sigma
                return out
            else:
                T_mat = t.t() @ t
                out =self.k(x1).evaluate()- kzx.t()@sol + T_mat#/self.sigma
                return out
        else:
            kzx_1 = self.k(Z, x1).evaluate()
            kzx_2 = self.k(Z, x2).evaluate()
            t= L.t() @ kzx_2
            t_ = kzx_1.t() @ L
            T_mat = t_ @ t
            if self.parametrize_Z:
                kzz = self.k(Z).evaluate()
                chol_z = torch.linalg.cholesky(kzz + self.eye * self.reg)
                sol = torch.cholesky_solve(kzx_2, chol_z)
            else:
                # sol = torch.cholesky_solve(kzx_2, self.inv_L)
                sol = self.inv_Z @ kzx_2
            out=self.k(x1, x2).evaluate() - kzx_1.t() @ sol +T_mat#/self.sigma
            return out

    def get_sigma_debug(self):
        with torch.no_grad():
            L = torch.tril(self.L)+self.eye*self.reg
            L=ensure_pos_diag(L)

            return L@L.t()

def get_hermite_weights(n):
    roots,weights = roots_hermite(n,False)
    return torch.tensor(roots).float(), torch.tensor(weights).float()

#TODO: FIX SCALING ISSUES

class GWI(torch.nn.Module):
    def __init__(self,N,m_q,m_p,r,reg=1e-1,sigma=1.0,APQ=False,empirical_sigma=1.0,x_s=250):
        super(GWI, self).__init__()
        self.r = r
        self.m_q = m_q
        self.sigma=sigma
        self.k=self.r.k
        self.reg = reg
        self.m_p=m_p
        self.m=self.r.Z.shape[0]
        self.x_s=x_s
        self.register_buffer('eye',reg*torch.eye(self.m))
        self.register_buffer('big_eye',100.*torch.eye(self.x_s))
        self.U_calculated = False
        self.N=N
        self.APQ = APQ
        self.log_empirical_sigma = np.log(empirical_sigma)
        self.const = (0.5*log2pi +self.log_empirical_sigma).item()

    def get_MPQ(self,batch_X=None):
        raise NotImplementedError

    def get_APQ(self,batch_X,Z_prime,T=None):
        X=batch_X
        rk_hat= 1/X.shape[0] * self.r(Z_prime,X)@self.k(X,Z_prime).evaluate()  #self.r.rk(X)/
        if T is not None:
            rk_hat = T*rk_hat
        eigs = torch.linalg.eigvals(rk_hat + self.big_eye)
        eigs = eigs.abs()
        eigs = eigs-self.big_eye.diag()
        eigs = eigs[eigs > 0]
        res = torch.sum(eigs**0.5)/self.x_s**0.5
        # self.calculate_V()
        return res

    def get_APQ_diagnose_xs(self,batch_X,Z_prime):
        X=batch_X
        rk_hat= 1/X.shape[0] * self.r(Z_prime,X)@self.k(X,Z_prime).evaluate()  #self.r.rk(X)/
        eigs = torch.linalg.eigvals(rk_hat + self.big_eye)
        eigs = eigs.abs()
        eigs = eigs-self.big_eye.diag()
        eigs = eigs[eigs > 0]

        return eigs.detach().cpu().numpy()

    def calc_hard_tr_term(self,X,Z_prime,T=None):
        mpq= self.get_APQ(X,Z_prime,T)
        p_trace = self.k(X.unsqueeze(1)).evaluate().mean()
        q_trace = self.r(X.unsqueeze(1)).mean()
        if T is not None:
            q_trace = q_trace*T
        mat = p_trace + q_trace -2*mpq#"*eig.diag().sum()
        return mat,-2*mpq,q_trace,p_trace

    def posterior_variance(self,X,T=None):
        with torch.no_grad():
            posterior = self.r(X.unsqueeze(1)).squeeze() +self.sigma
            if T is not None:
                posterior = posterior*T
        return posterior**0.5

    def likelihood_reg(self,y,X,T=None):
        pred = self.m_q(X)
        vec=y-pred
        tmp=torch.ones_like(y)*self.m_p
        reg = torch.mean((pred-tmp)**2) #M_Q - M_P
        sigma = self.sigma
        if T is not None:
            sigma = T*sigma
        return self.N*torch.mean(vec**2)/(2. *sigma),reg#+torch.sum(tmp.diag())/(2*self.sigma)

    def calc_NLL(self,y,X,T=None):
        pred = self.m_q(X)
        vec=(y-pred)**2
        rXX = self.r(X.unsqueeze(1)).squeeze(1) + self.sigma
        if T is not None:
            a=(0.5*torch.log(rXX*T)).sum()
            b=(0.5*vec/(rXX*T)).sum()
        else:
            a=(0.5*torch.log(rXX)).sum()
            b=(0.5*vec/rXX).sum()

        # print(a,b)
        return (a+b) + self.const*pred.shape[0]

    def get_loss(self,y,X,Z_prime,T=None):
        tot_trace,hard_trace,tr_Q,tr_P=self.calc_hard_tr_term(X,Z_prime,T)
        # print('MPQ: ', hard_trace)
        # print('Tr Q: ', tr_Q)
        # print('Tr P: ', tr_P)
        ll, reg= self.likelihood_reg(y,X,T)
        D = torch.relu((tot_trace + reg))**0.5 #this feels a bit broken?! a small trace term should make a small NLL???????
        sigma = self.sigma
        if T is not None:
            sigma = sigma*T
        log_loss = self.N*tr_Q / (2. * sigma) + ll
        return log_loss/X.shape[0],D

    def mean_forward(self,X):
        return self.m_q(X)

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)

class GVI_multi_classification(torch.nn.Module):
    def __init__(self,Z,N,m_q,m_p,k_list,r_list,reg=1e-3,sigma=1.0,eps=0.01,num_classes=10,APQ=False,x_s=100):
        super(GVI_multi_classification, self).__init__()
        self.N=N
        self.register_buffer('Z',Z)
        self.r = r_list #module list
        self.m_q = m_q #convnet
        self.sigma = sigma
        self.k = k_list #same prior or module list
        self.reg = reg
        self.m = r_list[0].Z.shape[0]
        self.x_s=x_s
        self.m_p = m_p #vector with empirical frequency? or just 0.5 i.e. don't make it so certain
        self.register_buffer('eye', reg * torch.eye(self.m))
        self.register_buffer('big_eye', 100. * torch.eye(self.x_s))
        self.U_calculated = False
        roots,weights=get_hermite_weights(50)
        self.register_buffer('gh_roots',roots.unsqueeze(0))
        self.register_buffer('gh_weights',weights.unsqueeze(0))
        self.dist = Normal(0,1)
        self.register_buffer('const_remove',torch.log(self.dist.cdf(2**0.5*self.gh_roots)+1e-3))
        self.register_buffer('const_divide',self.dist.cdf(2**0.5*self.gh_roots)+1e-3)
        self.register_buffer('log_1_minus_eps',torch.log(torch.tensor(1-eps)))
        self.register_buffer('log_1_over_eps',torch.log(torch.tensor(eps/(num_classes-1))))
        self.eps=eps
        self.num_classes = num_classes
        self.APQ = APQ

    def get_APQ(self,batch_X,Z_prime_flat):
        X=batch_X
        total_eig = 0.0
        for r in self.r:
            rk_hat = 1 / X.shape[0] * r(Z_prime_flat, X) @ self.k(X, Z_prime_flat).evaluate()  # self.r.rk(X)/
            eigs = torch.linalg.eigvals(rk_hat + self.big_eye)
            eigs = eigs.abs()
            eigs = eigs - self.big_eye.diag()
            eigs = eigs[eigs > 0]
            res = torch.sum(eigs ** 0.5) / self.x_s ** 0.5
            total_eig+=res
        return total_eig


    def get_MPQ(self,batch_X):
        raise NotImplementedError

    def calc_hard_tr_term(self,X,Z_prime_flat):
        hard_trace=self.get_APQ(X,Z_prime_flat)
        sq_trq_mat=[]
        total_r_trace = 0.0
        x_flattened = X.flatten(1).unsqueeze(1)
        total_p_trace = self.k(x_flattened).evaluate().mean()* len(self.r)
        for r in self.r:
            tmp = r(x_flattened).squeeze()
            sq_trq_mat.append(tmp)
            total_r_trace+=tmp.mean()
        sq_trq_mat=torch.stack(sq_trq_mat,dim=1)

        tot_trace = total_p_trace+total_r_trace -2*hard_trace
        return  tot_trace,hard_trace,total_p_trace,total_r_trace, sq_trq_mat

    def likelihood_reg(self,y,X):
        h = self.m_q(X)
        tmp=torch.ones_like(h)*self.m_p
        reg = torch.sum((h-tmp)**2)
        return h,reg

    def get_loss(self,y,X,Z_prime):
        Z_prime_flat = Z_prime.flatten(1)
        tot_trace,hard_trace,tr_P,tr_Q,sq_trq_mat = self.calc_hard_tr_term(X.flatten(1),Z_prime_flat)
        mean_pred,reg = self.likelihood_reg(y,X)
        trq_j = 2**0.5  * torch.gather(sq_trq_mat,1,y.unsqueeze(-1))  * self.gh_roots +torch.gather(mean_pred,1,y.unsqueeze(-1))
        cdf_term= (trq_j.unsqueeze(1)-mean_pred.unsqueeze(-1))/sq_trq_mat.unsqueeze(-1)
        full = torch.log(self.dist.cdf(cdf_term) +1e-3).sum(1)-self.const_remove
        S = torch.sum(full.exp()*self.gh_weights,-1)/sqrt_pi
        L=-(self.log_1_minus_eps*S+self.log_1_over_eps*(1-S))
        D = torch.relu(tot_trace+ reg) ** 0.5
        return L.sum(),D

    def mean_forward(self,X,T=None):
        m_q = self.m_q(X)
        sq_trq_mat = []
        x_flattened = X.flatten(1).unsqueeze(1)
        for r in self.r:
            var = r(x_flattened).squeeze()
            sq_trq_mat.append(var)
        sq_trq_mat = torch.stack(sq_trq_mat, dim=1)
        if T is not None:
            sq_trq_mat = sq_trq_mat*T

        trq_j = 2 ** 0.5 * sq_trq_mat.unsqueeze(-1) * self.gh_roots + m_q.unsqueeze(-1)
        cdf_term = (trq_j.unsqueeze(1) - m_q.unsqueeze(-1).unsqueeze(-1)) / sq_trq_mat.unsqueeze(-1).unsqueeze(-1)
        full = torch.log(self.dist.cdf(cdf_term) + 1e-3).sum(1) - self.const_remove
        S = torch.sum(full.exp() * self.gh_weights, -1) / sqrt_pi
        return (1 - self.eps) * S + (self.eps / (self.num_classes - 1)) * (1 - S)

    # def mean_pred(self,X,T=None):
    #     with torch.no_grad():
    #         m_q = self.m_q(X)
    #         sq_trq_mat = []
    #         x_flattened = X.flatten(1).unsqueeze(1)
    #         for r in self.r:
    #             var = r(x_flattened).squeeze()
    #             if T is not None:
    #                 sq_trq_mat.append(var * T)
    #             else:
    #                 sq_trq_mat.append(var)
    #         sq_trq_mat = torch.stack(sq_trq_mat, dim=1)
    #         trq_j = 2 ** 0.5 * sq_trq_mat.unsqueeze(-1) * self.gh_roots + m_q.unsqueeze(-1)
    #         cdf_term = (trq_j.unsqueeze(1) - m_q.unsqueeze(-1).unsqueeze(-1)) / sq_trq_mat.unsqueeze(-1).unsqueeze(-1)
    #         full = torch.log(self.dist.cdf(cdf_term) +1e-3).sum(1)-self.const_remove
    #         S = torch.sum(full.exp() * self.gh_weights, -1) / sqrt_pi
    #         return (1-self.eps)*S + (self.eps/(self.num_classes-1))*(1-S)

    def measure_similarity(self,X):
        l = self.k.base_kernel
        mat = l(X.flatten(1), self.Z.flatten(1)).evaluate()
        sim_max = torch.max(mat)
        sim_min = torch.min(mat)
        return sim_max,sim_min

    def mean_pred_prior(self,X,lim=1e-3,T=None):
        with torch.no_grad():
            l = self.k.base_kernel
            sim,_ = torch.max(l(X.flatten(1),self.Z.flatten(1)).evaluate(),dim=1)
            mask = (sim < lim).unsqueeze(-1)
            true_preds = self.mean_forward(X,T)
            prior_preds = torch.ones_like(true_preds)*self.m_p
            output = (~mask)*true_preds + mask*prior_preds
        return output

    def calc_NLL(self,y,X,T=None):
        mean_pred,reg = self.likelihood_reg(y,X)
        sq_trq_mat=[]
        x_flattened = X.flatten(1).unsqueeze(1)
        for r in self.r:
            var = r(x_flattened).squeeze()
            sq_trq_mat.append(var)
        sq_trq_mat = torch.stack(sq_trq_mat, dim=1)
        if T is not None:
            sq_trq_mat = sq_trq_mat*T


        trq_j = 2**0.5  * torch.gather(sq_trq_mat,1,y.unsqueeze(-1))  * self.gh_roots +torch.gather(mean_pred,1,y.unsqueeze(-1))
        cdf_term= (trq_j.unsqueeze(1)-mean_pred.unsqueeze(-1))/sq_trq_mat.unsqueeze(-1)
        full = torch.log(self.dist.cdf(cdf_term) +1e-3).sum(1)-self.const_remove
        S = torch.sum(full.exp()*self.gh_weights,-1)/sqrt_pi
        L=-torch.log((1-self.eps)*S+(self.eps/(self.num_classes-1))*(1-S)).sum()
        return L
    # def output(self):

        #preds_raw = m(X)
        #h = softmax
        # h_L = divide with diagonal of posterior
        #