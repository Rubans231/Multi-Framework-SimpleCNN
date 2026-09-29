import jax
import jax.numpy as jnp
import numpy as np

# jnp is very similar to regular np
x = jnp.arange(9.0).reshape(3, 3)
y = jnp.ones(3)

print(x @ y)
print(x.sum(axis=0))
print(x[1, :2])

# The arrayes are instances of jax.Array
print(isinstance(x, jax.Array))

print(jax.typeof(x))
