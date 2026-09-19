import torch
from .utils import vectorize
from torch.linalg import vector_norm


def _compute_norm(tensor, p, dim):
    """Compute norm using torch.norm for MPS devices, vector_norm otherwise.

    Args:
        tensor (torch.Tensor): input tensor
        p (float): norm order
        dim (int): dimension along which to compute norm

    Returns:
        torch.Tensor: computed norm
    """
    if tensor.device.type == 'mps':
        return torch.norm(tensor, p=p, dim=dim)
    else:
        return vector_norm(tensor, ord=p, dim=dim)


def energy_loss(x_true, x_est, beta=1, verbose=True):
    """Loss function based on the energy score.

    Args:
        x_true (torch.Tensor): iid samples from the true distribution of shape (data_size, data_dim)
        x_est (list of torch.Tensor): 
            - a list of length sample_size, where each element is a tensor of shape (data_size, data_dim) that contains one sample for each data point from the estimated distribution, or 
            - a tensor of shape (data_size*sample_size, response_dim) such that x_est[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size.
        beta (float): power parameter in the energy score.
        verbose (bool): whether to return two terms of the loss.

    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x_true = vectorize(x_true).unsqueeze(1)
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1)
        
    s1 = (_compute_norm(x_est - x_true, 2, dim=2) + EPS).pow(beta).mean()
    s2 = (torch.cdist(x_est, x_est, 2) + EPS).pow(beta).mean() * m / (m - 1)
    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)
    

def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=True, weights=None):
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x0 (torch.Tensor): an iid sample from the true distribution.
        x (torch.Tensor): an iid sample from the estimated distribution.
        xp (torch.Tensor): another iid sample from the estimated distribution.
        xp0 (torch.Tensor): another iid sample from the true distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
    
    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    if weights is None:
        weights = 1 / x0.size(0)
    if x0p is None:
        s1 = ((_compute_norm(x - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2 + ((_compute_norm(xp - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2
        s2 = ((_compute_norm(x - xp, 2, dim=1) + EPS).pow(beta) * weights).sum() 
        loss = s1 - s2/2
    else:
        x0p = vectorize(x0p)
        s1 = ((_compute_norm(x - x0, 2, dim=1) + EPS).pow(beta).sum() + (_compute_norm(xp - x0, 2, dim=1) + EPS).pow(beta).sum() +
              (_compute_norm(x - x0p, 2, dim=1) + EPS).pow(beta).sum() + (_compute_norm(xp - x0p, 2, dim=1) + EPS).pow(beta).sum()) / 4
        s2 = (_compute_norm(x - xp, 2, dim=1) + EPS).pow(beta).sum()
        s3 = (_compute_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta).sum() 
        loss = s1 - s2/2 - s3/2
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss

def gaussian_kernel_loss_two_sample(x0, x, xp, sigma=1.0, verbose=True):
    """Training loss based on the (negative) Gaussian Kernel Score.

    Kernel: k(a, b) = exp(-||a - b||^2 / (2 * sigma^2))

    Convention matches scoringrules.gksmv_ensemble (sigma=1 reproduces it exactly).
    Negatively oriented: lower is better, same as the energy loss.

    Two-sample estimator — structurally mirrors energy_loss_two_sample:
        loss = -E[k(X, y)] + 0.5 * E[k(X, X')]
    where both expectations are estimated from the two independent model draws
    x and xp. The constant 0.5*k(y,y)=0.5 is omitted (zero gradient).

    NOTE on sigma: scoringrules hardcodes sigma=1.  On 10-dim standardized
    residual paths ||x-y||^2 ~ 18-20, so exp(-9) ~ 1e-4 and the loss is
    near-degenerate (all terms collapse, near-zero gradients). This is expected
    and intentional when sigma=1 is used as a baseline experiment. Increase
    sigma (e.g. via median heuristic sqrt(median(||x-y||^2)/2)) for a
    non-degenerate training signal.

    Args:
        x0  (torch.Tensor): true responses,          shape (batch, dim).
        x   (torch.Tensor): model sample #1,         shape (batch, dim).
        xp  (torch.Tensor): model sample #2 (indep), shape (batch, dim).
        sigma (float): kernel bandwidth. Default 1.0 matches scoringrules.
        verbose (bool): if True return (loss, E_k_xy, E_k_xxp); else loss only.

    Returns:
        torch.Tensor: scalar loss, or 3-element tensor [loss, s1, s2] if verbose.
    """
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)

    two_s2 = 2.0 * sigma * sigma

    # -E[k(X, y)]: average both draws for variance reduction (mirrors s1 in ES)
    k_x_x0 = torch.exp(-(x - x0).pow(2).sum(dim=1) / two_s2)
    k_xp_x0 = torch.exp(-(xp - x0).pow(2).sum(dim=1) / two_s2)
    s1 = 0.5 * (k_x_x0.mean() + k_xp_x0.mean())  # E[k(X, y)]

    # +0.5 * E[k(X, X')]: independent pair -> unbiased cross term (mirrors s2 in ES)
    k_x_xp = torch.exp(-(x - xp).pow(2).sum(dim=1) / two_s2)
    s2 = k_x_xp.mean()  # E[k(X, X')]

    loss = -s1 + 0.5 * s2

    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss