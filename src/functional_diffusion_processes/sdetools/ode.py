import abc
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKeyArray

from ..sdetools.base_sde import SDE
from ..sdetools.sde_utils import construct_b_and_r, rand_phase
from ..utils.common import batch_mul


class RectifiedODE(SDE, abc.ABC):
    """Heat RectifiedODE class representing a deterministic ODE for functional rectified flow.

    This subclass implements:
      - score_fn that returns 0,
      - a marginal probability corresponding to the process:
            x_t = (1-t) * X0 + t * X1,
        where X1 is a sample of noise.
      - an sde method that returns the drift computed from this interpolation and a zero diffusion.

    The get_psm and prior_sampling interfaces are kept.
    """

    def __init__(self, sde_config):
        """Initialize the HeatSubVPODE instance.

        Args:
            sde_config: A configuration object/dictionary containing at least:
                - shape: the shape of the data.
                - (optional) b: a parameter used for get_psm (default 1.0).
                - (optional) r: a parameter used for get_psm (default 1.0).
        """
        super().__init__(sde_config)
        self.shape = sde_config.shape
        self.is_unidimensional = len(self.shape) == 1
        self.x_norm = sde_config.x_norm
        self.energy_norm = sde_config.energy_norm
        self.b, self.r = construct_b_and_r(self.x_norm, self.energy_norm, shape=self.shape)
        self.sigma = 1.0
        self.band_width = 1.0

    def _k(self, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute the k function value at a given time t.

        Args:
            t (jnp.ndarray): The time at which to compute the k function.
            t0 (Optional[jnp.ndarray], optional): The starting time for computing k. Defaults to None.

        Returns:
            jnp.ndarray: The value of k at time t.
        """
        raise NotImplementedError("This method should not be called.")

    def get_psm(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the PSM (Power-Special-Matrix) at time t.

        This implementation is kept unchanged from the original.

        Args:
            t: The current time as a jnp.ndarray.

        Returns:
            A jnp.ndarray weighting the loss function.
        """
        if not self.is_unidimensional:
            t = jnp.expand_dims(t, axis=(-1))
        if self.sde_config.psm_choice == "fdp" or self.sde_config.psm_choice is None:
            psm = jnp.expand_dims(jnp.ones_like(t) * jnp.sqrt(self.b / self.r).reshape(1, *self.shape), -1)
        elif self.sde_config.psm_choice == "all_ones":
            psm = jnp.ones((t.shape[0], *self.shape, 1))
        elif Path(self.sde_config.psm_choice).is_file():
            psm = jnp.array(np.load(self.sde_config.psm_choice))
            psm = jnp.expand_dims(psm, axis=(0, -1))
            assert psm.shape == (1, *self.shape, 1), f"Shape mismatch: expected {(1, *self.shape, 1)}, got {psm.shape}"
            psm = jnp.broadcast_to(psm, (t.shape[0], *self.shape, 1))
            assert psm.shape == (
                t.shape[0],
                *self.shape,
                1,
            ), f"Shape mismatch: expected {(t.shape[0], *self.shape, 1)}, got {psm.shape}"
        else:
            raise NotImplementedError("Unimplemented PSM type")
        return psm

    def score_fn(
        self,
        y_corrupted: jnp.ndarray,
        y_reconstructed: jnp.ndarray,
        t: jnp.ndarray,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Compute the score function.

        For this ODE, the score function is set to 0.

        Args:
            y_corrupted: The current (corrupted) state.
            y_reconstructed: The reconstructed (or original) state.
            t: The current time.
            rng: Optional random key.

        Returns:
            An array of zeros with the same shape as y_corrupted.
        """
        raise NotImplementedError("This method should not be called.")

    def prior_sampling(
        self, rng: jax.random.PRNGKey, shape: Tuple[int, ...], t0: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Sample from the prior distribution.

        This implementation is kept unchanged and simply samples standard Gaussian noise.

        Args:
            rng: The random number generator key.
            shape: The shape of the sample to be generated.
            t0: Optional initial time.

        Returns:
            A sample from a standard normal distribution.
        """
        b, g, c = shape
        t = jnp.ones((b, 1))
        t = jnp.expand_dims(t, axis=(-1))
        if t0 is not None:
            t0 = jnp.expand_dims(t0, axis=(-1))
        # Generate the noise as a Gaussian random variable.
        z = jax.random.normal(key=rng, shape=shape)
        z_freq = self.fourier_transform(state=z.reshape(b, *self.shape, c))

        rng, step_rng = jax.random.split(rng)
        phase = rand_phase(step_rng, z_freq.shape)
        fact = (self.r / self.b).reshape(1, *self.b.shape, 1)
        fact = jnp.broadcast_to(fact, (b, *self.shape, 1))  # * (1 - self._k(t, t0)) ** 2
        sigma = jnp.sqrt(jnp.abs(fact)).reshape(b, *self.shape, 1)
        z = jnp.real(self.inverse_fourier_transform(state=batch_mul(batch_mul(sigma, z_freq), phase))).reshape(b, g, c)
        return z
        """nx, ny = self.shape

        x = jnp.linspace(0.0, 1.0, num=nx)
        y = jnp.linspace(0.0, 1.0, num=ny)
        # Use 'ij' indexing so that X has shape (nx, ny).
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        # Stack to get a grid of points with shape (nx, ny, 2)
        XY = jnp.stack([X, Y], axis=-1)

        # Flatten the grid: from (nx, ny, 2) to (n_points, 2)
        grid_points = XY.reshape(-1, 2)
        n_points = grid_points.shape[0]

        # Compute pairwise Euclidean distances between grid points.
        # diff: shape (n_points, n_points, 2)
        diff = grid_points[:, None, :] - grid_points[None, :, :]
        # Euclidean norm: shape (n_points, n_points)
        r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))

        # Matern-3/2 kernel:
        #   k(r) = sigma² * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
        sqrt3 = math.sqrt(3.0)
        K = self.sigma**2 * (1.0 + sqrt3 * r / self.band_width) * jnp.exp(-sqrt3 * r / self.band_width)

        # Add a small jitter for numerical stability.
        jitter = 1e-6
        K += jitter * jnp.eye(n_points)

        # Compute the Cholesky decomposition.
        L = jnp.linalg.cholesky(K)  # shape: (n_points, n_points)

        # Interpret the provided shape as (batch_size, channels).
        batch_size, g, channels = shape
        assert g == n_points

        # Sample independent standard normal variates: shape (batch_size, n_points, channels)
        z = jax.random.normal(rng, shape=(batch_size, n_points, channels))

        # Impose the GP covariance using the Cholesky factor.
        # 'ij,bjc->bic' multiplies L (over grid points) with z.
        gp_sample = jnp.einsum('ij,bjc->bic', L, z)

        return gp_sample.reshape(batch_size, n_points, channels)"""

    def marginal_prob(
        self,
        rng: jax.random.PRNGKey,
        x: jnp.ndarray,
        t: jnp.float32,
        t0: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the marginal probability for the ODE.

        This implements the process:
            x_t = (1-t) * x + t * noise,
        where `x` is the original state (X0) and `noise` is sampled from a standard normal.

        Args:
            rng: The random number generator key.
            x: The original state X0.
            t: The interpolation time (should be between 0 and 1).
            t0: Unused in this ODE; kept for interface consistency.

        Returns:
            A tuple (x_t, noise_std) where:
              - x_t is the interpolated state.
              - noise_std is set to t.
        """
        b, g, c = x.shape
        # x_freq = self.fourier_transform(state=x.reshape(b, *self.shape, c))
        # t = jnp.expand_dims(t, axis=-1)
        # if t0 is not None:
        #    t0 = jnp.expand_dims(t0, axis=-1)
        # k_t = self._k(t, t0).reshape(b, *self.shape, 1)
        # x_freq = batch_mul(k_t, x_freq)
        # x0t = jnp.real(self.inverse_fourier_transform(state=x_freq)).reshape(b, g, c)
        # phase = rand_phase(rng, (b, *self.shape, c))
        # fact = self.r / self.b * (1 - self._k(t, t0)) ** 2
        # print(self.r.shape, self.b.shape, t.shape, fact.shape) # (32, 32) (32, 32) (64, 1, 1) (64, 32, 32)
        # sigma = jnp.sqrt(jnp.abs(fact)).reshape(b, *self.shape, 1)
        # noise_std = batch_mul(sigma, phase)
        t = t.reshape(b, *[1] * (len(self.shape) + 1))
        t = jnp.broadcast_to(t, (b, *self.shape, 1))
        x0t = x.reshape(b, *self.shape, c) * t
        return x0t, 1.0 - t

    def sde(
        self,
        y_corrupted: jnp.ndarray,
        t: jnp.float32,
        rng: Optional[jax.random.PRNGKey] = None,
        y_reconstructed: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the drift and diffusion terms for the ODE at time t.

        Given the marginal process
            y(t) = (1-t) * y_reconstructed + t * noise,
        the time derivative is
            dy/dt = (noise - y_reconstructed) = (y_corrupted - y_reconstructed) / t,
        where we assume y_reconstructed corresponds to X0 and y_corrupted to x_t.

        Diffusion is set to zero.

        Args:
            y_corrupted: The current state (i.e. (1-t)*X0 + t*noise).
            t: The current time.
            rng: Optional random key (unused here).
            y_reconstructed: The original state X0; required for computing the drift.

        Returns:
            A tuple (drift, diffusion) with diffusion = 0.
        """
        raise NotImplementedError("This method should not be called.")

    def get_reverse_noise(self, rng: PRNGKeyArray, shape: Tuple[int, ...]) -> jnp.ndarray:
        raise NotImplementedError("This method should not be called.")

    def diffuse(
        self, rng: PRNGKeyArray, x: jnp.ndarray, t: jnp.ndarray, t0: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("This method should not be called.")
