import torch
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def sim_sin_curve(n=1000,noise=0.5,seed=1):
    torch.manual_seed(seed)
    x = torch.linspace(-10,10,n)
    y=torch.sin(x)+torch.randn_like(x)*noise + 0.1*x
    return x.unsqueeze(-1),y.unsqueeze(-1)

def sim_sin_curve_2(n=1000,noise=0.5,seed=1):
    torch.manual_seed(seed)
    x = torch.linspace(-10,10,n)
    y=torch.sin(x)+torch.randn_like(x)*noise + 0.1*x**2
    return x.unsqueeze(-1),y.unsqueeze(-1)

def sim_sin_curve_3(n=1000,noise=0.5,seed=1):
    torch.manual_seed(seed)
    x = torch.linspace(-1,1,n)
    y = torch.sin(x * 3 * 3.14) + 0.3 * torch.cos(x * 9 * 3.14) + 0.5 * torch.sin(x * 7 * 3.14)
    y = y +  torch.randn_like(x)*noise
    return x.unsqueeze(-1),y.unsqueeze(-1)


if __name__ == '__main__':

    x,y=sim_sin_curve(1000,0.4)
    sns.scatterplot(x.squeeze(),y.squeeze())
    plt.show()