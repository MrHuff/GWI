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
    v = torch.clamp(v,min=1e-3)
    mask = torch.diag(torch.ones_like(v))
    L = mask * torch.diag(v) + (1. - mask) * L
    return L

class ls_init(torch.nn.Module):
    def __init__(self,k,y,Z,sigma,its=150):
        super(ls_init, self).__init__()
        self.its = its
        self.register_buffer('eye',torch.eye(Z.shape[0])*sigma)
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
            L=torch.linalg.cholesky(self.k(self.Z).evaluate()+self.eye*10)
            loss= -self.objective(L)
            print(loss)
            loss.backward()
            opt.step()
        return self.k

class r_param_cholesky_scaling(torch.nn.Module):
    def __init__(self,k,Z,X,sigma,reg=1e-3,scale_init=0.0,parametrize_Z=False):
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
        with torch.no_grad():
            kx= self.k(self.Z,self.X).evaluate()
            self.kzz =self.k(self.Z).evaluate()
            # print('kzz rank: ',torch.linalg.matrix_rank(self.kzz))
            L = torch.linalg.cholesky(torch.inverse(self.kzz+kx@kx.t()/self.sigma+self.eye))

        if not self.parametrize_Z:
            self.kzz_inv = torch.inverse(self.kzz+self.eye*self.reg)
        self.L = torch.nn.Parameter(L)
        # geotorch.

    #L getting fucked up.
    def forward(self,x1,x2=None):
        L = torch.tril(self.L)+self.eye*self.reg
        L=torch.sigmoid(ensure_pos_diag(L))
        Z= self.Z
        if x2 is None:
            kzx = self.k(Z, x1).evaluate()
            t= L.t() @ kzx

            if self.parametrize_Z:
                kzz = self.k(Z).evaluate()
                chol_z = torch.linalg.cholesky(kzz+self.eye*self.reg)
                sol = torch.cholesky_solve(kzx,chol_z)
            else:
                sol =self.kzz_inv @ kzx
            if len(t.shape)==3:
                T_mat = t.permute(0,2,1) @ t
                out = self.k(x1).evaluate()- kzx.permute(0,2,1)@sol + T_mat/self.sigma
                # return out.clamp(min=1e-3)
                return out
            else:
                T_mat = t.t() @ t
                out =(self.k(x1).evaluate()- kzx.t()@sol + T_mat/self.sigma)
                # out=ensure_pos_diag(out)
                return out
        else:
            kzx_1 = self.k(Z, x1).evaluate()
            kzx_2 = self.k(Z, x2).evaluate()
            t= L.t() @ kzx_2
            t_ = kzx_1 @ self.L
            if self.parametrize_Z:
                kzz = self.k(Z).evaluate()
                chol_z = torch.linalg.cholesky(kzz + self.eye * self.reg)
                sol = torch.cholesky_solve(kzx_2, chol_z)
            else:
                sol =self.kzz_inv @ kzx_2
            return  (self.k(x1,x2).evaluate()- kzx_1.t()@sol +  t_ @ t/self.sigma)

    def get_sigma_debug(self):
        with torch.no_grad():
            L = torch.tril(self.L)+self.eye*self.reg
            # L=ensure_pos_diag(L)
            L = torch.sigmoid(ensure_pos_diag(L))

            return L@L.t()

    def rk(self,X):
        L = torch.tril(self.L)+self.eye*self.reg
        # L=ensure_pos_diag(L)
        L=torch.sigmoid(ensure_pos_diag(L))

        Z= self.Z
        kzx = self.k(Z, X).evaluate()
        mid= L.t()@kzx
        if self.parametrize_Z:
            kzz = self.k(Z).evaluate()
            return (kzz@L@mid@kzx.t())
        else:
            return (self.kzz@L@mid@kzx.t())

def get_hermite_weights(n):
    roots,weights = roots_hermite(n,False)
    return torch.tensor(roots).float(), torch.tensor(weights).float()

#TODO: FIX SCALING ISSUES

class GWI(torch.nn.Module):
    def __init__(self,N,m_q,m_p,k,r,reg=1e-3,sigma=1.0,APQ=False):
        super(GWI, self).__init__()
        self.r = r
        self.m_q = m_q
        self.sigma=sigma
        self.k=k
        self.reg = reg
        self.m_p=m_p
        self.m=self.r.Z.shape[0]
        self.register_buffer('eye',reg*torch.eye(self.m))
        self.register_buffer('big_eye',100.*torch.eye(self.m))
        self.U_calculated = False
        self.N=N
        self.APQ = APQ

    def get_MPQ(self,batch_X=None):
        raise NotImplementedError

    def get_APQ(self,batch_X=None):
        X=batch_X
        rk_hat=self.r.rk(X)/X.shape[0]
        eigs = torch.linalg.eigvals(rk_hat + self.big_eye)
        eigs = eigs.abs()
        eigs = eigs-self.big_eye.diag()
        eigs = eigs[eigs > 0]
        res = torch.sum(eigs**0.5)/self.m**0.5
        # self.calculate_V()
        return res

    def calc_hard_tr_term(self,X=None):
        mpq= self.get_APQ(X)
        p_trace = self.k(X.unsqueeze(1)).evaluate().mean()
        q_trace = self.r(X.unsqueeze(1)).mean()
        mat = p_trace + q_trace -2*mpq#"*eig.diag().sum()
        return mat,-2*mpq,q_trace,p_trace

    def posterior_variance(self,X):
        with torch.no_grad():
            posterior = self.r(X.unsqueeze(1)).squeeze()
        return posterior**0.5

    def likelihood_reg(self,y,X):
        pred = self.m_q(X)
        vec=y-pred
        tmp=torch.ones_like(y)*self.m_p
        reg = torch.mean((pred-tmp)**2) #M_Q - M_P
        return self.N*torch.mean(vec**2)/(2. *self.sigma),reg#+torch.sum(tmp.diag())/(2*self.sigma)

    def calc_NLL(self,y,X):
        pred = self.m_q(X)
        vec=(y-pred)**2
        rXX=self.r(X.unsqueeze(1)).squeeze(1)+self.sigma
        return 0.5*(torch.log(rXX)+vec/rXX + log2pi).sum()

    def get_loss(self,y,X):
        tot_trace,hard_trace,tr_Q,tr_P=self.calc_hard_tr_term(X)
        # print('MPQ: ', hard_trace)
        # print('Tr Q: ', tr_Q)
        # print('Tr P: ', tr_P)
        ll, reg= self.likelihood_reg(y,X)
        D = torch.relu((tot_trace + reg))**0.5
        log_loss = self.N*tr_Q / (2. * self.sigma) + ll
        return log_loss/X.shape[0],D

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)

class GVI_multi_classification(torch.nn.Module):
    def __init__(self,N,m_q,m_p,k_list,r_list,reg=1e-3,sigma=1.0,eps=0.01,num_classes=10,APQ=False):
        super(GVI_multi_classification, self).__init__()
        self.N=N
        self.r = r_list #module list
        self.m_q = m_q #convnet
        self.sigma = sigma
        self.k = k_list #same prior or module list
        self.reg = reg
        self.m = r_list[0].Z.shape[0]
        self.m_p = m_p #vector with empirical frequency? or just 0.5 i.e. don't make it so certain
        self.register_buffer('eye', reg * torch.eye(self.m))
        self.register_buffer('big_eye', 100. * torch.eye(self.m))
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

    def get_APQ(self,batch_X):
        X=batch_X
        total_eig = 0.0
        for r in self.r:
            rk_hat = r.rk(X)/X.shape[0]
            eigs = torch.linalg.eigvals(rk_hat + self.big_eye)
            eigs = eigs.abs()
            eigs = eigs - self.big_eye.diag()
            eigs = eigs[eigs > 0]
            res = torch.sum(eigs ** 0.5) / self.m ** 0.5
            total_eig+=res
        return total_eig


    def get_MPQ(self,batch_X):
        raise NotImplementedError

    def calc_hard_tr_term(self,X):
        hard_trace=self.get_APQ(X)
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

    def get_loss(self,y,X):
        tot_trace,hard_trace,tr_P,tr_Q,sq_trq_mat = self.calc_hard_tr_term(X.flatten(1))
        mean_pred,reg = self.likelihood_reg(y,X)
        trq_j = 2**0.5  * torch.gather(sq_trq_mat,1,y.unsqueeze(-1))  * self.gh_roots +torch.gather(mean_pred,1,y.unsqueeze(-1))
        cdf_term= (trq_j.unsqueeze(1)-mean_pred.unsqueeze(-1))/sq_trq_mat.unsqueeze(-1)
        full = torch.log(self.dist.cdf(cdf_term) +1e-3).sum(1)-self.const_remove
        S = torch.sum(full.exp()*self.gh_weights,-1)/sqrt_pi
        L=-(self.log_1_minus_eps*S+self.log_1_over_eps*(1-S))
        D = torch.relu(tot_trace+ reg) ** 0.5
        return L.sum(),D

    def mean_pred(self,X):
        with torch.no_grad():
            m_q = self.m_q(X)
            sq_trq_mat = []
            x_flattened = X.flatten(1).unsqueeze(1)
            for r in self.r:
                sq_trq_mat.append(r(x_flattened).squeeze())
            sq_trq_mat = torch.stack(sq_trq_mat, dim=1)
            trq_j = 2 ** 0.5 * sq_trq_mat.unsqueeze(-1) * self.gh_roots + m_q.unsqueeze(-1)
            cdf_term = (trq_j.unsqueeze(1) - m_q.unsqueeze(-1).unsqueeze(-1)) / sq_trq_mat.unsqueeze(-1).unsqueeze(-1)
            full = torch.log(self.dist.cdf(cdf_term) +1e-3).sum(1)-self.const_remove
            S = torch.sum(full.exp() * self.gh_weights, -1) / sqrt_pi
            return (1-self.eps)*S + (self.eps/(self.num_classes-1))*(1-S)

    def calc_NLL(self,y,X):
        mean_pred,reg = self.likelihood_reg(y,X)
        sq_trq_mat=[]
        x_flattened = X.flatten(1).unsqueeze(1)
        for r in self.r:
            sq_trq_mat.append(r(x_flattened).squeeze())
        sq_trq_mat=torch.stack(sq_trq_mat,dim=1)
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