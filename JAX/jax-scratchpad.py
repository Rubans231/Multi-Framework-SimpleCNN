import jax
import jax.numpy as jnp
import numpy as np

x = jnp.arange(9.0).reshape(3, 3)
y = jnp.ones(3)

print(x @ y)
print(x.sum(axis=0))
print(x[1, :2])
