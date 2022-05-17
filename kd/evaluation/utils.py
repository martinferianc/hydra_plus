import torch.nn.functional as F
import torch

@torch.no_grad()
def classification_uncertainty(samples):
    # Decompose the output with repsect to predictive, aleatorc and epistemic uncertainties
    # Shape: (batch_size, N, num_classes)
    if len(samples.shape) == 2:
        samples = samples.unsqueeze(1)
    softmax = F.softmax(samples, dim=2)
    mean_softmax = torch.mean(softmax, dim=1)
    predictive = torch.sum(-mean_softmax * torch.log(mean_softmax+1e-8), dim=1)
    aleatoric = torch.mean(
        torch.sum(-softmax * torch.log(softmax+1e-8), dim=2), dim=1)
    epistemic = predictive - aleatoric
    return predictive, aleatoric, epistemic


@torch.no_grad()
def regression_uncertainty(samples):
    # Decompose the output with repsect to predictive, aleatorc and epistemic uncertainties
    # Shape: (batch_size, samples, mean+var)
    if len(samples.shape) == 2:
        samples = samples.unsqueeze(1)
    mean, var = samples[:, :, 0], samples[:, :, 1].exp()
    mean_var = mean.var(dim=1)
    # Replace nans with zeros, this is if the sample size is 1
    mean_var[torch.isnan(mean_var)] = 0.0
    predictive = torch.sqrt(mean_var + var.mean(dim=1))
    aleatoric = torch.sqrt(var.mean(dim=1))
    epistemic = torch.sqrt(mean.var(dim=1))
    return predictive, aleatoric, epistemic
