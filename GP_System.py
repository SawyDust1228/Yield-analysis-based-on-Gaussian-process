import torch
import gpytorch
from botorch.distributions import Kumaraswamy
from gpytorch.kernels import InducingPointKernel


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dim):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=dim) + gpytorch.kernels.LinearKernel(num_dimensions=dim))
        # self.base_covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.MaternKernel(ard_num_dims=dim) + gpytorch.kernels.LinearKernel(num_dimensions=dim))
        # self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:1280, :],
        #                                         likelihood=likelihood)
        self.c1 = torch.rand(dim, dtype=torch.float32) * 3 + 0.1
        self.c0 = torch.rand(dim, dtype=torch.float32) * 3 + 0.1


    def forward(self, x):
        k = Kumaraswamy(concentration1=self.c1, concentration0=self.c0)
        x = k.icdf(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyGP():
    def __init__(self, train_x, train_y, dim,training_iter):
        self.train_x = train_x
        self.train_y = train_y
        self.dim = dim
        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(train_x, train_y, self.likelihood, dim=dim)
        self.training_iter = training_iter

    # this is for running the notebook in our testing framework

    def train(self):
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            # print(output)
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                i + 1, self.training_iter, loss.item(),
                self.model.likelihood.noise.item()
            ))
            optimizer.step()

    def test(self, test_x):
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
        return observed_pred
