from typing import Sequence

import haiku as hk
import jax.numpy as jnp
from haiku._src.typing import Initializer

hk.initializers.Initializer = Initializer

class ZeroOneInitializer(hk.initializers.Initializer):
  def __init__(self) -> None:
      super().__init__()

  def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
    assert len(shape) == 1
    z_bias = -jnp.ones(shape[0] // 3, dtype)
    other_bias = jnp.zeros(2*shape[0] // 3, dtype)

    return jnp.concatenate([z_bias, other_bias])
