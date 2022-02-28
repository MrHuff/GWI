import torch
from gpytorch.kernels import RBFKernel

class r_param(torch.nn.Module):
    def __init__(self,k,Z,d=10):
        super(r_param, self).__init__()
        self.k = k
        # self.register_buffer('Z',Z)
        self.Z = torch.nn.Parameter(Z)
        self.L = torch.nn.Parameter(0.5*torch.randn(self.Z.shape[0],d))
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

# U matrix is eigenvector matrix of k, which is associated with P
# V matrix is eivenmatrix matrix of r, which is associated with Q

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


        # self.choose_m()


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


        lamb_U_cut=(1./(self.n*lamb_U_cut**0.5)).unsqueeze(-1)
        self.register_buffer('M',lamb_U_cut@lamb_U_cut.t())
        return U_slim,torch.sum(lamb_U_cut),lamb_U_cut@lamb_U_cut.t()

    def calculate_V(self): #Pretty sure this is just the Nyström approximation to the inverse haha!
        tmp=self.r(self.Z)
        # lamb_V,V= torch.symeig(tmp, eigenvectors=True)
        # mask = lamb_V>1e-2
        self.r_mat = tmp
        return tmp, tmp.diag().sum() #lamb_V[mask].sum() #"*(1+1./(2.*self.sigma)) #*(1./(2.*self.sigma)+1./self.m)

    def full_V_inv(self):
        with torch.no_grad():
            self.r_inv_full =torch.inverse(self.r(self.X).evaluate())

    def get_MPQ(self):
        # U,tr_P,M=self.calculate_U()
        # print(U,tr_P,M)
        rk_hat=self.r(self.Z,self.X)@self.k(self.X,self.Z).evaluate()
        # rk_hat =rk_hat.evaluate()
        V_hat_mu,trace_Q= self.calculate_V()
        one= rk_hat@self.U
        res = torch.cholesky_solve(one,V_hat_mu)
        res = one.t()@res * self.M
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

    def mean_pred(self,X):
        with torch.no_grad():
            return self.m_q(X)




