import jax
import jax.numpy as jnp
import numpy as np

# jnp is very similar to regular np
# np and jnp can be used interchangeably
# it’s common to use np for data loading and lightweight host-side manipulation and jnp for number crunching on accelerators
x = jnp.arange(9.0).reshape(3, 3)
y = jnp.ones(3)

print(x @ y)
print(x.sum(axis=0))
print(x[1, :2])

# The arrayes are instances of jax.Array
print(isinstance(x, jax.Array))

print(jax.typeof(x))

# numpy is mutable while jnp isn't
# x[0] = 10 is possible in numpy while not possible in jnp
# instead jax allows for "at" which returns a new Array with the update applied

y = x.at[0].set(10)
print(x)  # remains same
print(y)

# alongside set there is also add, multiply, min and max
# code without mutation is easier to transform, optimize and parallelize as the jit apparently covers the updates by itself

print(x.at[4].add(67))
print(x.at[:3].max(5))

# JAX defaults to 32-bit which is good for accelerators, while np makes 64bit by def
# JAX actually disables 64-bit altogether and can only be enabled through config change
print(jnp.array([1.0, 5.0]).dtype)

# Numpy provides int32 and float16 to float64 but jax keeps the float16

d = np.arange(3, dtype=np.int32)
i = np.ones(3, dtype=np.float16)
print((d + i).dtype)  # numpy
print((jnp.asarray(d) + jnp.asarray(i)).dtype)  # JAX
