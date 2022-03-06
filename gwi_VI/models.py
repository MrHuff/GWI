import torch
import tqdm
from gpytorch.kernels import RBFKernel
from scipy.special import roots_hermite
import numpy as np
from torch.distributions.normal import Normal


CDF_APPROX_COEFF=1.65451
sqrt_pi=np.pi**0.5

class r_param(torch.nn.Module):
    def __init__(self,k,Z,init=0.5,d=10,scale_init=0.0):
        super(r_param, self).__init__()
        self.k = k
        # self.register_buffer('Z',Z)
        self.Z = torch.nn.Parameter(Z)
        self.L = torch.nn.Parameter(init*torch.randn(self.Z.shape[0],d))
        self.scale = torch.nn.Parameter(torch.ones(1)*scale_init)

    def forward(self,x1,x2=None):
        if x2 is None:
            t= self.L.t() @ self.k(self.Z,x1).evaluate()
            return torch.exp(self.scale)*(self.k(x1).evaluate() + t.t() @ t)
            # return (self.k(x1).evaluate() + t.t() @ t)
        else:
            t= self.L.t() @ self.k(self.Z,x2).evaluate()
            t_ = self.k(x1,self.Z).evaluate() @ self.L
            return  torch.exp(self.scale)*(self.k(x1,x2).evaluate() + t_ @ t)
            # return  (self.k(x1,x2).evaluate() + t_ @ t)

class ls_init(torch.nn.Module):
    def __init__(self,k,y,Z,sigma,its=50):
        super(ls_init, self).__init__()
        self.its = its
        self.register_buffer('eye',torch.eye(Z.shape[0])*sigma)
        self.register_buffer('Z',Z)
        self.register_buffer('Y',y)

        self.k=k

    def objective(self,L):
        inv,_=torch.triangular_solve(self.Y,L)
        return -torch.diag(L).log().sum()-0.5*torch.sum(inv**2)

    def pre_train(self):
        opt = torch.optim.Adam(self.k.parameters(),lr=1e-2)
        for i in tqdm.tqdm(range(self.its)):
            opt.zero_grad()
            L=torch.linalg.cholesky(self.k(self.Z).evaluate()+self.eye*1e-1)
            loss= self.objective(L)
            loss.backward()
            opt.step()

class r_param_cholesky(torch.nn.Module):
    def __init__(self,k,Z,X,sigma,reg=1e-3,scale_init=0.0):
        super(r_param_cholesky, self).__init__()
        self.k = k
        self.Z = torch.nn.Parameter(Z)
        self.scale = torch.nn.Parameter(torch.ones(1)*scale_init)
        self.register_buffer('eye',torch.eye(Z.shape[0]))
        self.register_buffer('X',X)
        self.reg=reg
        self.sigma=sigma


    def init_L(self):
        with torch.no_grad():
            kx= self.k(self.Z,self.X).evaluate()
            kzz =self.k(self.Z).evaluate()
            L = torch.inverse(torch.linalg.cholesky(kzz+kx@kx.t()/self.sigma+1./self.sigma**0.5*self.eye))
            # print(L)
        self.L = torch.nn.Parameter(L)


    def forward(self,x1,x2=None):
        L = torch.tril(self.L) + self.eye * self.reg
        if x2 is None:
            t= L.t() @ self.k(self.Z,x1).evaluate()
            if len(t.shape)==3:
                T_mat = t.permute(0,2,1) @ t
            else:
                T_mat = t.t() @ t
            return torch.exp(self.scale)*(self.k(x1).evaluate() + T_mat)
            # return (self.k(x1).evaluate() + t.t() @ t)
            # return  t.t() @ t #torch.exp(self.scale)*t.t() @ t  + self.k(x1).evaluate()
        else:
            t= L.t() @ self.k(self.Z,x2).evaluate()
            t_ = self.k(x1,self.Z).evaluate() @ self.L
            return  torch.exp(self.scale)*(self.k(x1,x2).evaluate() + t_ @ t)
            # return  (self.k(x1,x2).evaluate() + t_ @ t)
            # return  t_ @ t #self.k(x1,x2).evaluate() + torch.exp(self.scale)* t_ @ t


# U matrix is eigenvector matrix of k, which is associated with P
# V matrix is eivenmatrix matrix of r, which is associated with Q


def get_hermite_weights(n):
    roots,weights = roots_hermite(n,False)
    return torch.tensor(roots).float(), torch.tensor(weights).float()

#Approximate function

class GWI(torch.nn.Module):
    def __init__(self,m_q,m_p,k,r,Z,reg=1e-3,sigma=1.0,APQ=False):
        super(GWI, self).__init__()
        self.r = r
        self.m_q = m_q
        self.sigma=sigma
        self.k=k
        self.reg = reg
        self.m_p=m_p
        self.m=Z.shape[0]
        self.register_buffer('eye',reg*torch.eye(self.m))
        self.register_buffer('big_eye',100.*torch.eye(self.m))
        self.register_buffer('Z', Z)
        self.U_calculated = False
        self.APQ = APQ

    def calculate_U(self):
        self.k_hat=self.k(self.Z).evaluate()
        self.register_buffer('tr_P', torch.sum(self.k_hat.diag()))
        if not self.APQ:
            lamb_U,U= torch.linalg.eigh(self.k_hat+self.big_eye)
            lamb_U = lamb_U-torch.diag(self.big_eye)
            mask = lamb_U>1e-2
            lamb_U_cut = lamb_U[mask]
            self.eff_m = torch.sum(mask).item()
            print('eff m ', self.eff_m)
            U_slim = U[:,mask]
            self.register_buffer('U',U_slim)
            # self.register_buffer('tr_P',torch.sum(lamb_U_cut))

            lamb_U_cut=(1./(lamb_U_cut**0.5)).unsqueeze(-1)
            self.register_buffer('M',lamb_U_cut@lamb_U_cut.t())
            return U_slim,torch.sum(lamb_U_cut),lamb_U_cut@lamb_U_cut.t()

    def calculate_V(self): #Pretty sure this is just the Nyström approximation to the inverse haha!
        tmp=self.r(self.Z)
        # lamb_V,V= torch.symeig(tmp, eigenvectors=True)
        # mask = lamb_V>1e-2
        self.r_mat = tmp
        return tmp, tmp.diag().sum() #lamb_V[mask].sum() #"*(1+1./(2.*self.sigma)) #*(1./(2.*self.sigma)+1./self.m)

    def get_MPQ(self,batch_X=None):
        if not self.U_calculated:
            self.U_calculated=True
            self.calculate_U()
        # U,tr_P,M=self.calculate_U()
        # print(U,tr_P,M)
        X=batch_X
        rk_hat=self.r(self.Z,X)@self.k(X,self.Z).evaluate()
        # rk_hat =rk_hat.evaluate()
        V_hat_mu,trace_Q= self.calculate_V()
        one= rk_hat@self.U
        res = torch.linalg.solve(V_hat_mu,one)
        res = one.t()@res * self.M /(X.shape[0])**2
        return res,trace_Q,self.tr_P#/self.m

    def get_APQ(self,batch_X=None):

        if not self.U_calculated:
            self.U_calculated=True
            self.calculate_U()
            self.k_hat_inv = torch.inverse(self.k_hat)

        if batch_X is not None:
            X=batch_X
        else:
            X=self.X


        rk_hat=self.r(self.Z,X)@self.k(X,self.Z).evaluate()
        # rk_hat =rk_hat.evaluate()
        V_hat_mu,trace_Q= self.calculate_V()

        a,_ = torch.solve(rk_hat,V_hat_mu)
        b,_ =  self.k_hat_inv@rk_hat.t()
        res = b@a /(batch_X.shape[0]**2)
        return res,trace_Q,self.tr_P#/self.m


    def calc_hard_tr_term(self,X=None):
        if self.APQ:
            mpq,trace_Q,trace_P= self.get_APQ(X)
        else:
            mpq,trace_Q,trace_P= self.get_MPQ(X)
        eig=torch.linalg.eigvalsh(mpq)
        eig = eig[eig>0]
        # print(eig)

        return -2*torch.sum(eig**0.5),trace_Q,trace_P

    def posterior_variance(self,X):
        with torch.no_grad():
            # tmp= self.r(X,self.Z)#.evaluate()
            # right = torch.cholesky_solve(tmp.t(),self.r_mat)
            # # posterior = self.r(X).evaluate()-(tmp@right)
            posterior = self.r(X)
        return torch.diag(posterior)**0.5

    def likelihood_reg(self,y,X):
        pred = self.m_q(X)
        vec=y-pred
        tmp=torch.ones_like(y)*self.m_p
        reg = torch.sum((pred-tmp)**2)**0.5
        # tmp=self.r(self.Z)
        # v,_= torch.symeig(tmp,True)
        # v = v[v>0]
        return torch.sum(vec**2)/(2*self.sigma),reg#+torch.sum(tmp.diag())/(2*self.sigma)

    def get_loss(self,y,X):
        hard_trace,tr_Q,tr_P=self.calc_hard_tr_term(X)
        ll, reg= self.likelihood_reg(y,X)
        D = (hard_trace + tr_Q + reg)
        log_loss = tr_Q / (2. * self.sigma) + ll
        return log_loss/X.shape[0],D

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)

class GVI_binary_classification(GWI):
    def __init__(self,m_q,m_p,k,r,Z,reg=1e-3,sigma=1.0,APQ=False):
        super(GVI_binary_classification, self).__init__(m_q,m_p,k,r,Z,reg,sigma,APQ)
        roots,weights=get_hermite_weights(100)
        self.register_buffer('gh_roots',roots.unsqueeze(0))
        self.register_buffer('gh_weights',weights.unsqueeze(0))

    def likelihood_reg(self,y,X):
        pred = self.m_q(X)
        tmp=torch.ones_like(y)*self.m_p
        reg = torch.sum((pred-tmp)**2)**0.5
        # tmp=self.r(self.Z)
        # v,_= torch.symeig(tmp,True)
        # v = v[v>0]
        return pred,reg#+torch.sum(tmp.diag())/(2*self.sigma)

    def get_loss(self,y,X):
        hard_trace,tr_Q,tr_P=self.calc_hard_tr_term(X)
        mean_pred,reg= self.likelihood_reg(y,X)
        D = (hard_trace + tr_Q + reg)
        binary_input = (2.*tr_Q)**0.5*self.gh_roots+mean_pred
        loss = torch.relu(binary_input)-y*binary_input+torch.log1p(binary_input.abs().exp())
        log_loss = (loss*self.gh_weights).sum(1).sum()
        return log_loss/X.shape[0],D

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)._sigmoid()

class GVI_multi_classification(torch.nn.Module):
    def __init__(self,m_q,m_p,k_list,r_list,Z,reg=1e-3,sigma=1.0,eps=0.1):
        super(GVI_multi_classification, self).__init__()
        self.eps=eps
        self.r = r_list #module list
        self.m_q = m_q #convnet
        self.sigma = sigma
        self.k = k_list #same prior or module list
        self.reg = reg
        self.m_p = m_p #vector with empirical frequency? or just 0.5 i.e. don't make it so certain
        self.m = Z.shape[0]
        self.register_buffer('eye', reg * torch.eye(self.m))
        self.register_buffer('big_eye', 100. * torch.eye(self.m))
        self.register_buffer('Z', Z)
        self.U_calculated = False
        roots,weights=get_hermite_weights(100)
        self.register_buffer('gh_roots',roots.unsqueeze(0))
        self.register_buffer('gh_weights',weights.unsqueeze(0))
        self.dist = Normal(0,1)
        self.register_buffer('const_remove',torch.log(self.dist.cdf(2**0.5*self.gh_roots)+1e-3))

    def calculate_U(self):
        self.k_hat=self.k(self.Z).evaluate()
        self.register_buffer('tr_P', torch.sum(self.k_hat.diag()))

        lamb_U,U= torch.linalg.eigh(self.k_hat+self.big_eye)
        lamb_U = lamb_U-torch.diag(self.big_eye)
        mask = lamb_U>1e-2
        lamb_U_cut = lamb_U[mask]
        self.eff_m = torch.sum(mask).item()
        print('eff m ', self.eff_m)
        U_slim = U[:,mask]
        self.register_buffer('U',U_slim)
        # self.register_buffer('tr_P',torch.sum(lamb_U_cut))

        lamb_U_cut=(1./(lamb_U_cut**0.5)).unsqueeze(-1)
        self.register_buffer('M',lamb_U_cut@lamb_U_cut.t())
            # return U_slim,torch.sum(lamb_U_cut),lamb_U_cut@lamb_U_cut.t()

    def calculate_U_list(self):
        self.k_hat=[]
        self.tr_P=[]
        for i,k in enumerate(self.k):
            tmp=k(self.Z).evaluate()
            self.tr_P+=tmp.diag().sum()
            self.k_hat.append(tmp)

        self.U=[]
        self.M=[]
        if not self.APQ:
            for k_hat in self.k_hat:
                lamb_U,U= torch.symeig(k_hat+self.big_eye, eigenvectors=True)
                lamb_U = lamb_U-torch.diag(self.big_eye)
                mask = lamb_U>1e-2
                lamb_U_cut = lamb_U[mask]
                # self.eff_m = torch.sum(mask).item()
                # print('eff m ', self.eff_m)
                U_slim = U[:,mask]
                self.U.append(U_slim)
                lamb_U_cut=(1./(lamb_U_cut**0.5)).unsqueeze(-1)
                self.M.append(lamb_U_cut@lamb_U_cut.t())
        self.U=torch.stack(self.U,dim=0)
        self.M=torch.stack(self.M,dim=0)

    def calculate_V(self): #Pretty sure this is just the Nyström approximation to the inverse haha!
        self.r_mat=[]
        tr_Q=0.0
        for i,r in enumerate(self.r):
            tmp=r(self.Z)
            self.r_mat.append(tmp)
            tr_Q+=tmp.diag().sum()
        return self.r_mat,tr_Q  #lamb_V[mask].sum() #"*(1+1./(2.*self.sigma)) #*(1./(2.*self.sigma)+1./self.m)

    def get_MPQ(self,batch_X):
        if not self.U_calculated:
            self.U_calculated=True
            self.calculate_U()
        X=batch_X
        rk_hat=[]
        for k,r in zip(self.k,self.r):
            rk_hat.append(r(self.Z,X)@k(X,self.Z).evaluate())

        rk_hat_batch= torch.stack(rk_hat,dim=0)
        V_hat_mu,trace_Q= self.calculate_V()
        V_hat_batch = torch.stack(V_hat_mu,dim=0)
        one=rk_hat_batch@self.U#rk_hatself.U
        res = torch.linalg.solve(V_hat_batch,one)
        res = one.permute(0,2,1)@res * self.M /(X.shape[0])**2


        return res,trace_Q,self.tr_P#/self.m

    def calc_hard_tr_term(self,X):
        mpq,tr_Q_mat,tr_P_mat=self.get_MPQ(X)
        # eig,v =  torch.symeig(mpq, eigenvectors=True)
        eig =  torch.linalg.eigvalsh(mpq)
        eig = eig[eig>0]
        return  -2*torch.sum(eig**0.5),tr_Q_mat,tr_P_mat

    def likelihood_reg(self,y,X):
        pred = self.m_q(X)
        h = torch.softmax(pred,dim=1)
        tmp=torch.ones_like(pred)*self.m_p
        reg = torch.sum((h-tmp)**2)**0.5
        return h,reg

    def get_loss(self,y,X):
        hard_trace,tr_Q,tr_P = self.calc_hard_tr_term(X.flatten(1))
        mean_pred,reg = self.likelihood_reg(y,X)
        sq_trq_mat=[]
        x_flattened = X.flatten(1).unsqueeze(1)
        for r in self.r:
            sq_trq_mat.append(r(x_flattened).squeeze())
        sq_trq_mat=torch.stack(sq_trq_mat,dim=1)
        trq_j = 2**0.5  * torch.gather(sq_trq_mat,1,y.unsqueeze(-1))  * self.gh_roots +torch.gather(mean_pred,1,y.unsqueeze(-1))
        cdf_term= (trq_j.unsqueeze(1)-mean_pred.unsqueeze(-1))/sq_trq_mat.unsqueeze(-1)
        # full = torch.log(self.dist.cdf(cdf_term) +1e-3).sum(1)-self.const_remove
        full = -torch.log(1+torch.exp(-cdf_term*CDF_APPROX_COEFF)).sum(1)-self.const_remove
        S = torch.sum(full.exp()*self.gh_weights,-1)/sqrt_pi
        L=-torch.log(S)-torch.log(1-S)
        D = (hard_trace + tr_Q + reg)
        return L.sum(),D

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X).softmax(dim=1)

    # def output(self):

        #preds_raw = m(X)
        #h = softmax
        # h_L = divide with diagonal of posterior
        #