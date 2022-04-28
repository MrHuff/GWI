import torch
import tqdm
from gpytorch.kernels import RBFKernel
from scipy.special import roots_hermite
import numpy as np
from torch.distributions.normal import Normal

CDF_APPROX_COEFF=1.65451
sqrt_pi=np.pi**0.5
log2pi=np.log(np.pi*2)

def ensure_pos_diag(L):
    v=torch.diag(L)
    # print(torch.diag(L))
    v = torch.clamp(v,min=1e-6)
    mask = torch.diag(torch.ones_like(v))
    L = mask * torch.diag(v) + (1. - mask) * L
    return L

class ls_init(torch.nn.Module):
    def __init__(self,k,y,Z,sigma,its=25):
        super(ls_init, self).__init__()
        self.its = its
        self.register_buffer('eye',torch.eye(Z.shape[0]))
        self.register_buffer('Z',Z)
        self.register_buffer('Y',y)

        self.k=k

    def objective(self,L):
        inv=torch.cholesky_solve(self.Y,L)
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

class r_param_cholesky_scaling(torch.nn.Module):
    def __init__(self,k,Z,X,sigma,reg=1e-1,scale_init=1.0,parametrize_Z=False):
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
        self.reg=reg
        self.sigma=sigma

    def init_L(self):
        num_instable = False
        try:
            with torch.no_grad():
                kx = self.k(self.Z, self.X).evaluate()
                self.kzz = self.k(self.Z).evaluate()
                print(self.kzz)
                L = torch.inverse(torch.linalg.cholesky(self.kzz+kx@kx.t()/self.sigma+self.eye))
        except Exception as e:
            print(e)
            print('-----------------------------------------------------CHOLESKY ERROR--------------------------------------------')
            num_instable = True
        if num_instable:
            torch.cuda.empty_cache()
            with torch.no_grad():
                self.kzz = self.k(self.Z).evaluate()
                L= torch.randn_like(self.kzz)*0.1
        if not self.parametrize_Z:
            self.kzz_inv = torch.inverse(self.kzz+self.eye*self.reg)
        self.L = torch.nn.Parameter(L)
    #L getting fucked up.
    def forward(self,x1,x2=None):
        L = torch.tril(self.L)+self.eye*self.reg
        L = ensure_pos_diag(L)
        # L=torch.sigmoid(ensure_pos_diag(L))
        Z= self.Z
        if x2 is None:
            kzx = self.k(Z, x1).evaluate()
            t= L.t() @ kzx #L\cdot k(Z,X)

            if self.parametrize_Z:
                kzz = self.k(Z).evaluate()
                chol_z = torch.linalg.cholesky(kzz+self.eye*self.reg)
                sol = torch.cholesky_solve(kzx,chol_z)
            else:
                sol =self.kzz_inv @ kzx
            if len(t.shape)==3: #t=[L^T k(Z,X_i),L^T k(Z,X_{i+1}),]
                T_mat = t.permute(0,2,1) @ t
                out = self.k(x1).evaluate()- kzx.permute(0,2,1)@sol + T_mat/self.sigma
                return out
                #lower bound the scale, interpret as nuancing the prior
                #Too certain if NLL is shit.

            else:
                T_mat = t.t() @ t
                out =self.k(x1).evaluate()- kzx.t()@sol + T_mat/self.sigma
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
                sol =self.kzz_inv @ kzx_2
            out=self.k(x1, x2).evaluate() - kzx_1.t() @ sol +T_mat/ self.sigma
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

    def get_APQ(self,batch_X,Z_prime):
        X=batch_X
        rk_hat= 1/X.shape[0] * self.r(Z_prime,X)@self.k(X,Z_prime).evaluate()  #self.r.rk(X)/
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

    def calc_hard_tr_term(self,X,Z_prime):
        mpq= self.get_APQ(X,Z_prime)
        p_trace = self.k(X.unsqueeze(1)).evaluate().mean()
        q_trace = self.r(X.unsqueeze(1)).mean()
        mat = p_trace + q_trace -2*mpq#"*eig.diag().sum()
        return mat,-2*mpq,q_trace,p_trace

    def posterior_variance(self,X,T=None):
        with torch.no_grad():
            posterior = self.r(X.unsqueeze(1)).squeeze() +self.sigma
            if T is not None:
                posterior = posterior*T
        return posterior**0.5

    def likelihood_reg(self,y,X):
        pred = self.m_q(X)
        vec=y-pred
        tmp=torch.ones_like(y)*self.m_p
        reg = torch.mean((pred-tmp)**2) #M_Q - M_P
        return self.N*torch.mean(vec**2)/(2. *self.sigma),reg#+torch.sum(tmp.diag())/(2*self.sigma)

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

    def get_loss(self,y,X,Z_prime):
        tot_trace,hard_trace,tr_Q,tr_P=self.calc_hard_tr_term(X,Z_prime)
        # print('MPQ: ', hard_trace)
        # print('Tr Q: ', tr_Q)
        # print('Tr P: ', tr_P)
        ll, reg= self.likelihood_reg(y,X)
        D = torch.relu((tot_trace + reg))**0.5 #this feels a bit broken?! a small trace term should make a small NLL???????
        log_loss = self.N*tr_Q / (2. * self.sigma) + ll
        # print(ll.mean())
        # print((self.N*tr_Q / (2. * self.sigma)).mean())
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