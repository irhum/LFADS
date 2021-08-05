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

# minimal implementation, only works inside hk.transform
# TODO: check if drop_mask needs to be passed into inner function when jitted
def dynamic_unroll_recur_drop(core, input_sequence, initial_state, drop_mask):
  def scan_f(prev_state, inputs):
    outputs, next_state = core(inputs * drop_mask, prev_state)
    return next_state, outputs 

  final_state, output_sequence = hk.scan(scan_f, initial_state, input_sequence)
  return output_sequence, final_state
