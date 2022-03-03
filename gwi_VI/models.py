import torch
import tqdm
from gpytorch.kernels import RBFKernel
from scipy.special import roots_hermite



class r_param(torch.nn.Module):
    def __init__(self,k,Z,init=0.5,d=10):
        super(r_param, self).__init__()
        self.k = k
        # self.register_buffer('Z',Z)
        self.Z = torch.nn.Parameter(Z)
        self.L = torch.nn.Parameter(init*torch.randn(self.Z.shape[0],d))
        self.scale = torch.nn.Parameter(-torch.ones(1))

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
    def __init__(self,k,y,Z,sigma,its):
        super(ls_init, self).__init__()
        self.its = its
        self.register_buffer('eye',torch.eye(Z.shape[0])*sigma)
        self.register_buffer('Z',torch.eye(Z))
        self.register_buffer('Y',y)

        self.k=k
    @staticmethod
    def objective(L):
        inv=torch.triangular_solve(y,L)
        return -torch.diag(L).log().sum()-0.5*torch.sum(inv**2)

    def pre_train(self):
        opt = torch.optim.Adam(self.k.parameters(),lr=1e-2)
        for i in tqdm.tqdm(range(self.its)):
            opt.zero_grad()
            L=torch.cholesky(self.k(self.Z).evaluate)
            loss= ls_init.objective(L)
            loss.backward()
            opt.step()

class r_param_cholesky(torch.nn.Module):
    def __init__(self,k,Z,X,sigma,reg=1e-2):
        super(r_param_cholesky, self).__init__()
        self.k = k
        self.Z = torch.nn.Parameter(Z)
        self.scale = torch.nn.Parameter(-torch.ones(1))
        self.register_buffer('eye',torch.eye(Z.shape[0]))
        self.reg=reg
        self.sigma=sigma
        with torch.no_grad():
            kx= self.k(Z,X).evaluate()
            kzz =self.k(Z).evaluate()
            L = torch.cholesky(torch.inverse(kzz+kx@kx.t()/sigma-self.eye))
        self.L = torch.nn.Parameter(L)


    def forward(self,x1,x2=None):
        L = torch.tril(self.L) + self.eye * self.reg
        if x2 is None:
            t= L.t() @ self.k(self.Z,x1).evaluate()
            return torch.exp(self.scale)*(self.k(x1).evaluate() + t.t() @ t)
            # return (self.k(x1).evaluate() + t.t() @ t)
        else:
            t= L.t() @ self.k(self.Z,x2).evaluate()
            t_ = self.k(x1,self.Z).evaluate() @ self.L
            return  torch.exp(self.scale)*(self.k(x1,x2).evaluate() + t_ @ t)
            # return  (self.k(x1,x2).evaluate() + t_ @ t)


# U matrix is eigenvector matrix of k, which is associated with P
# V matrix is eivenmatrix matrix of r, which is associated with Q


def get_hermite_weights(n):
    roots,weights = roots_hermite(n,False)
    return torch.tensor(roots).float(), torch.tensor(weights).float()

#Approximate function

class GWI(torch.nn.Module):
    def __init__(self,m_q,m_p,X,k,r,Z,reg=1e-3,sigma=1.0):
        super(GWI, self).__init__()
        self.r = r
        self.m_q = m_q
        self.sigma=sigma
        self.k=k
        self.reg = reg
        self.register_buffer('X',X)
        self.m_p=m_p
        self.n,self.d=X.shape
        self.m=Z.shape[0]
        self.register_buffer('eye',reg*torch.eye(self.m))
        self.register_buffer('big_eye',100.*torch.eye(self.m))
        self.register_buffer('Z', Z)

    def choose_m(self):
        tmp=self.k(self.Z).evaluate()+self.big_eye
        lamb_U,U= torch.symeig(tmp, eigenvectors=True)
        lamb_U = lamb_U-torch.diag(self.big_eye)
        self.eff_m = torch.sum(lamb_U>1e-2).item()
        print('eff m ', self.eff_m)
        # self.register_buffer('eye',self.reg*torch.eye(self.m))
        # self.register_buffer('big_eye',100.*torch.eye(self.m))
        # if self.n>self.m:
        #     self.register_buffer('Z',self.X[torch.randperm(self.n)[:self.m],:])
        # else:
        #     self.Z=self.X

    def calculate_U(self):
        tmp=self.k(self.Z).evaluate()
        lamb_U,U= torch.symeig(tmp+self.big_eye, eigenvectors=True)
        lamb_U = lamb_U-torch.diag(self.big_eye)
        mask = lamb_U>1e-2
        lamb_U_cut = lamb_U[mask]
        self.eff_m = torch.sum(mask).item()
        print('eff m ', self.eff_m)
        U_slim = U[:,mask]
        self.register_buffer('U',U_slim)
        # self.register_buffer('tr_P',torch.sum(lamb_U_cut))
        self.register_buffer('tr_P',torch.sum(tmp.diag()))

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
        # U,tr_P,M=self.calculate_U()
        # print(U,tr_P,M)
        if batch_X is not None:
            X=batch_X
        else:
            X=self.X
        rk_hat=self.r(self.Z,X)@self.k(X,self.Z).evaluate()
        # rk_hat =rk_hat.evaluate()
        V_hat_mu,trace_Q= self.calculate_V()
        one= rk_hat@self.U
        res,_ = torch.solve(one,V_hat_mu)
        res = one.t()@res * self.M /(X.shape[0])**2
        return res,trace_Q,self.tr_P#/self.m

    def calc_hard_tr_term(self):
        mpq,trace_Q,trace_P= self.get_MPQ()
        eig,v =  torch.symeig(mpq, eigenvectors=True)
        eig = eig[eig>0]
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
        return torch.mean(vec**2)/(2*self.sigma),reg#+torch.sum(tmp.diag())/(2*self.sigma)

    def get_loss(self,y,X):
        hard_trace,tr_Q,tr_P=self.calc_hard_tr_term()
        ll, reg= self.likelihood_reg(y,X)
        D = (hard_trace + tr_Q + reg)
        log_loss = tr_Q / (2. * self.sigma) + ll
        return log_loss,D

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)

class GVI_binary_classification(GWI):
    def __init__(self,m_q,m_p,X,k,r,Z,reg=1e-3,sigma=1.0):
        super(GVI_binary_classification, self).__init__(m_q,m_p,X,k,r,Z,reg,sigma)
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
        hard_trace,tr_Q,tr_P=self.calc_hard_tr_term()
        mean_pred,reg= self.likelihood_reg(y,X)
        D = (hard_trace + tr_Q + reg)
        binary_input = (2.*tr_Q)**0.5*self.gh_roots+mean_pred
        loss = torch.relu(binary_input)-y*binary_input+torch.log1p(binary_input.abs().exp())
        log_loss = (loss*self.gh_weights).sum(1).mean()
        return log_loss,D



    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)._sigmoid()



