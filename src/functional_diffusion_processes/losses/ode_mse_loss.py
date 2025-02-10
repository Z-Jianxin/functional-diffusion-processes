import abc
from functools import partial
from typing import Any, Callable, TypeVar, Union

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.random import PRNGKeyArray
from omegaconf import DictConfig

from ..models import BaseMAML, BaseViT
from ..sdetools import SDE

# from ..utils.common import batch_mul

Params = FrozenDict[str, Any]
T = TypeVar("T")


class ODEMSELoss(abc.ABC):
    """MSE loss for the ODE setting where the model output is noise-data.

    The loss is computed as the mean squared error between the model's predicted noise
    and the actual noise that was used to corrupt the clean data via the process:
        x_t = (1-t)*x + t*ε,
    where ε ~ N(0, I).
    """

    def __init__(self, sde: SDE, loss_config: DictConfig) -> None:
        """Initializes the MSELoss instance with SDE object and loss configuration.

        Args:
            sde (SDE): An object representing the stochastic differential equation.
            loss_config (DictConfig): A configuration object holding parameters for loss computation.
        """
        self.sde = sde
        self.loss_config = loss_config

    def construct_loss_fn(self, model: Union[BaseMAML, BaseViT]) -> Callable:
        """Constructs a loss function for ODE training.

        Args:
            model (Union[BaseMAML, BaseViT]): The model for which the loss function is being constructed.

        Returns:
            Callable: A function to compute the MSE loss given the necessary inputs.
        """
        update_params_fn = model.make_update_params_fn()

        reduce_op = jnp.mean if self.loss_config.reduce_mean else lambda *args, **kwargs: jnp.sum(*args, **kwargs)

        @jax.jit
        @partial(jax.grad, argnums=1, has_aux=True)
        def loss_fn(
            rng: PRNGKeyArray, params: Params, step: int, batch_input: jnp.ndarray, batch_real: jnp.ndarray
        ) -> T:
            """Computes the MSE loss for ODE between the predicted noise and the target noise.

            Args:
                rng (PRNGKeyArray): A random number generator.
                params (Params): Parameters of the model.
                step (int): Current training step.
                batch_input (jnp.ndarray): Input data batch.
                batch_real (jnp.ndarray): Clean data batch.

            Returns:
                T: A tuple containing the gradient of the loss and a tuple of auxiliary outputs.
            """
            b, g, c = batch_real.shape

            # Sample a time t for each instance.
            rng, step_rng = jax.random.split(rng)
            t = jax.random.uniform(step_rng, (b, 1), minval=self.sde.sde_config.eps, maxval=self.sde.sde_config.T)
            if self.loss_config.use_scheduler:
                t = t * (1 - jnp.exp(-step / self.loss_config.scheduler_steps))
            t_new = jnp.reshape(t, (b, 1, 1))
            t_new = jnp.broadcast_to(t_new, (b, g, 1))
            if self.loss_config.normalize_time:
                t_new = t_new * 2 - 1

            batch_input = batch_input.at[:, :, -1:].set(t_new)
            shape = self.sde.sde_config.shape
            rng, step_rng = jax.random.split(rng)
            # noise = jax.random.normal(rng, (b, g, c))
            # noise_freq = self.sde.fourier_transform(state=noise.reshape(b, *shape, c))

            rng, step_rng = jax.random.split(rng)
            mean, std = self.sde.marginal_prob(step_rng, batch_real, t)

            noise = self.sde.prior_sampling(step_rng, (b, g, c))
            noise_std = std * noise.reshape(b, *shape, c)
            # jnp.real(self.sde.inverse_fourier_transform(batch_mul(std, noise_freq)).reshape(b, g, c))
            batch_corrupted = mean + noise_std
            if self.loss_config.y_input:
                batch_input = batch_input.at[:, :, len(shape) : len(shape) + c].set(batch_corrupted.reshape(b, g, c))

            psm = self.sde.get_psm(t)
            new_rng, model_output, loss_inner = update_params_fn(rng, params, batch_input, batch_corrupted, psm)
            prediction = model_output.reshape(b, *shape, c)
            batch_corrupted = batch_corrupted.reshape(b, *shape, c)
            prediction = prediction.reshape(b, *shape, c)
            # batch_real = (batch_real-noise).reshape(b, *shape, c)
            batch_real = batch_real.reshape(b, *shape, c)

            if self.loss_config.frequency_space:
                prediction_freq = self.sde.fourier_transform(state=prediction)
                target_freq = self.sde.fourier_transform(state=batch_real)
                # psm = self.sde.get_psm(t)
                # squared_loss = jnp.abs(prediction_freq * psm - target_freq * psm) ** 2
                squared_loss = jnp.abs(prediction_freq - target_freq) ** 2
            else:
                # psm = self.sde.get_psm(t)
                # squared_loss = jnp.abs(prediction * psm - batch_real * psm) ** 2
                squared_loss = jnp.abs(prediction - batch_real) ** 2

            losses = reduce_op(squared_loss.reshape(squared_loss.shape[0], -1), axis=-1)
            loss = jnp.mean(losses) / c
            return loss, (new_rng, loss, loss_inner, prediction, batch_corrupted, batch_real)

        return loss_fn
