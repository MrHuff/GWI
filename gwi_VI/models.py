import torch
from gpytorch.kernels import RBFKernel


# U matrix is eigenvector matrix of k, which is associated with P
# V matrix is eivenmatrix matrix of r, which is associated with Q


class GWI(torch.nn.Module):
    def __init__(self,m_q,m_p,X,k,r,reg=1e-3,sigma=1.0,m=1000):
        super(GWI, self).__init__()
        self.r = r
        self.m_q = m_q
        self.sigma=sigma
        self.k=k

        self.register_buffer('X',X)
        self.m_p=m_p
        self.n,self.d=X.shape
        self.m=m
        self.register_buffer('eye',reg*torch.eye(self.m))
        if self.n>self.m:
            self.register_buffer('Z',self.X[torch.randperm(self.n)[:m],:])
        else:
            self.Z=self.X


    def calculate_U(self):
        tmp=self.k(self.Z).evaluate()+ self.eye
        lamb_U,U=torch.eig(tmp,eigenvectors=True)
        self.register_buffer('U',U)
        lamb_U=1./(self.m*lamb_U[:,0]**0.5)
        self.register_buffer('M',lamb_U@lamb_U.t())

    def calculate_V(self): #Pretty sure this is just the Nyström approximation to the inverse haha!
        tmp=self.r(self.Z).evaluate() + self.eye
        self.r_mat = tmp

        # mu_V,V=torch.eig(tmp,eigenvectors=True)
        # V_hat_mu = 1/mu_V[:,0]**0.5 * V
        V_hat_mu=torch.inverse(tmp)
        return V_hat_mu,torch.diag(tmp).sum()*(1./(2.*self.sigma+1./self.m))

    def full_V_inv(self):
        with torch.no_grad():
            self.r_inv_full =torch.inverse(self.r(self.X).evaluate())

    def get_MPQ(self):
        rk_hat=self.r(self.Z,self.X)@self.k(self.X,self.Z)
        rk_hat =rk_hat.evaluate()
        V_hat_mu,trace_Q= self.calculate_V()
        one= rk_hat@self.U
        res = V_hat_mu@one
        res = one.t()@res * self.M
        return res,trace_Q

    def calc_hard_tr_term(self):
        mpq,trace_Q = self.get_MPQ()
        eig,v = torch.eig(mpq,eigenvectors=True)
        e=eig[:,0]
        mask=e>1e-2

        return torch.sum(e[mask]),trace_Q

    def invert_r_mat(self):
        self.r_inv = torch.inverse(self.r_mat)

    def posterior_variance(self,X):
        with torch.no_grad():
            tmp= self.r(X,self.Z).evaluate()
            posterior = self.r(X).evaluate()-(tmp@(self.r_inv))@(tmp.t())
        return torch.diag(posterior)

    def likelihood_reg(self,y,X):
        pred = self.m_q(X)
        vec=y-pred
        tmp=torch.ones_like(y)*self.m_p
        reg = torch.sum((pred-tmp)**2)
        return torch.mean(vec**2)/self.sigma,reg

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)




