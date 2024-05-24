import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Callable

DEFAULT_RESSHIFT_PARAMS = {
    "a": 1.1113,
    "alpha_T_prime": 2.3177576065063477,
    "kappa": 2.0,
    "T": 15,
}


def alpha_fn(t, a, alpha_T_prime, T):
  lamb = np.log(1 + alpha_T_prime / a) 
  return a * (np.exp(lamb * t) - 1) 

def eta_fn(t, a, alpha_T_prime, T): 
  lamb = np.log(1 + alpha_T_prime / a) 
  return a * ((np.exp(lamb * t) - 1) / lamb - t)

class BaseSampler:
  def __init__(
    self,
    *,
    ae,
    x0_pred_fn,
    a=DEFAULT_RESSHIFT_PARAMS["a"],
    T=DEFAULT_RESSHIFT_PARAMS["T"],
    alpha_T_prime=DEFAULT_RESSHIFT_PARAMS["alpha_T_prime"],
    kappa=DEFAULT_RESSHIFT_PARAMS["kappa"],
    device="cpu",
  ):
    self.ae = ae.to(device)
    self.x0_pred_fn = x0_pred_fn.to(device)
    self.alpha_fn = lambda t: alpha_fn(t, a, alpha_T_prime, T)
    self.eta_fn = lambda t: eta_fn(t, a, alpha_T_prime, T)

    self.T = T
    self.kappa = kappa

    self.device = device

    self.f = lambda alpha, e0, score:\
              alpha * e0 - 0.5 * alpha * self.kappa**2 * score

  def _get_score_with_x0(self, x, step_idx, x0, y0):
    eta = self._etas[step_idx]
    return -(x - x0 * (1 - eta) - eta * y0) / (self.kappa**2 * eta)

  def _ode_step(self, x, step_idx, y0, lq):
    pass

  def __call__(self, timesteps, lq):
    self._timesteps = timesteps.to(self.device)
    self._alphas = self.alpha_fn(timesteps).to(self.device)
    self._etas = self.eta_fn(timesteps).to(self.device)

    lq = lq.to(self.device)
    lq_up = F.interpolate(lq, scale_factor=4., mode='bicubic')
    
    y0 = self.ae.encode(lq_up)
    y0_noised = self._prior_sample(y0)
    
    x = y0_noised.to(self.device)
    for step_idx in range(len(timesteps) - 1):
      x = self._ode_step(x, step_idx, y0, lq)
    return self.ae.decode(x)

  def _prior_sample(self, y):
      eta_end = self._etas[0]
      return y + torch.randn_like(y) * eta_end**0.5 * self.kappa


class EulerSampler(BaseSampler):
    def _ode_step(self, x, step_idx, y0, lq):
        alpha = self._alphas[step_idx]
        eta = self._etas[step_idx]

        std = (self.kappa**2 * eta + 1)**0.5
        timestep = torch.tensor([self._timesteps[step_idx]] * len(x), dtype=x.dtype, device=x.device)
        x0 = self.x0_pred_fn(x / std, timestep * self.T - 1, lq)

        e0 = y0 - x0
        score = self._get_score_with_x0(x, step_idx, x0, y0)

        h = self._timesteps[step_idx + 1] - self._timesteps[step_idx]
        x_next= x + h * self.f(alpha, e0, score)
        return x_next