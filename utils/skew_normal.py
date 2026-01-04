from collections import namedtuple

import numpy as np
from scipy.stats import norm

from xgboost_distribution.compat import linalg_solve
from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import MAX_EXPONENT, MIN_EXPONENT

# Parameter bounds for log-scale
MIN_LOG_SCALE = MIN_EXPONENT / 2
MAX_LOG_SCALE = MAX_EXPONENT / 2

Params = namedtuple("Params", ("loc", "scale", "skew"))

class SkewNormal(BaseDistribution):
    """Skew-Normal distribution for real-valued targets.

    PDF:
        f(x) = 2 / scale * phi((x-loc)/scale) * Phi(skew * (x-loc)/scale)
    where phi/CDF are the standard normal PDF/CDF.
    """

    @property
    def params(self):
        return Params._fields

    def check_target(self, y):
        # no restriction: supports all real values
        return

    def starting_params(self, y):
        # loc ~ mean, scale ~ log(std), skew~0 (reduces to Normal)
        return Params(
            loc=np.mean(y).astype(np.float32),
            scale=np.log(np.std(y)).astype(np.float32),
            skew=np.zeros_like(y, dtype=np.float32)
        )

    def _safe_params(self, params):
        # extract and clip scale
        loc = params[:, 0]
        log_scale = np.clip(params[:, 1], a_min=MIN_LOG_SCALE, a_max=MAX_LOG_SCALE)
        skew = params[:, 2]
        return loc, log_scale, skew

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        loc, log_scale, skew = self._safe_params(params)
        scale = np.exp(log_scale)
        u = (y - loc) / scale
        phi_u = norm.pdf(u)
        Phi_u = norm.cdf(skew * u)

        # log-likelihood: log(2/scale) + log(phi_u) + log(Phi_u)
        # Derivatives:
        # ∂loc: (phi')/phi chain + skew-term; placeholder below
        g_loc = (loc - y) / (scale**2)  # TODO: replace with skew-normal derivative
        g_scale = 1 - (u**2)           # TODO: replace with skew-normal derivative
        g_skew = (phi_u * u) / Phi_u   # TODO: replace with skew-normal derivative

        # Hessians placeholders (diagonal)
        h_loc = 1 / (scale**2)         # TODO
        h_scale = 2 * (u**2)           # TODO
        h_skew = (phi_u * u / Phi_u)**2 # TODO

        grads = np.stack([g_loc, g_scale, g_skew], axis=1)
        hesses = np.stack([h_loc, h_scale, h_skew], axis=1)

        if natural_gradient:
            # Solve F^{-1} * grad for each sample
            fisher = np.zeros((len(y), 3, 3), dtype="float32")
            # Approximate diagonal Fisher; user can refine
            fisher[:, 0, 0] = 1 / (scale**2)
            fisher[:, 1, 1] = 2
            fisher[:, 2, 2] = 1  # placeholder
            grads = linalg_solve(fisher, grads)
            # use constant hessians
            hesses = np.ones_like(hesses)

        return grads, hesses

    def loss(self, y, params):
        loc, scale, skew = self.predict(params)
        # negative log-pdf
        u = (y - loc) / scale
        return "SkewNormal-NLL", -np.log(2/scale * norm.pdf(u) * norm.cdf(skew * u))

    def predict(self, params):
        loc, log_scale, skew = self._safe_params(params)
        scale = np.exp(log_scale)
        return Params(loc=loc, scale=scale, skew=skew)
