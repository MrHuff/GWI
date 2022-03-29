import tqdm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from utils.dataloaders import *

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class gp_svi_baseline(torch.nn.Module):
    def __init__(self,train_x,train_y,train_params,VI_params):

        super(gp_svi_baseline, self).__init__()
        self.train_params=train_params
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.n,self.d = train_x.shape
        m=int(round(self.n**0.5))
        inducing_points = train_x[torch.randperm(self.n)[:m],:]
        self.model = GPModel(inducing_points)
        dataset = general_custom_dataset(train_x,train_y)
        dataset.set('train')
        self.dl = custom_dataloader(dataset,batch_size=train_params['bs'],shuffle=True)
        self.device = train_params['device']
        self.train_y = train_y

    def train_model(self):
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)
        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.train_y.shape[0])
        for i in tqdm.tqdm(range(self.train_params['epochs'])):
            for i, (x_batch,x_cat, y_batch) in enumerate(tqdm.tqdm(self.dl)):
                x_batch = x_batch.to(self.device).squeeze()
                y_batch = y_batch.to(self.device).squeeze()
                if not isinstance(x_cat,list):
                    x_cat = x_cat.to(self.device)
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

    def eval_model(self,x_test):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model(x_test)
        lower, upper = observed_pred.confidence_region()
        mean = observed_pred.mean
        return mean,lower,upper
#
#
# model.eval()
# likelihood.eval()
# means = torch.tensor([0.])
# with torch.no_grad():
#     for x_batch, y_batch in test_loader:
#         preds = model(x_batch)
#         means = torch.cat([means, preds.mean.cpu()])
# means = means[1:]