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

# indexing is similar to np but exception raising doesn't exist, For example:
h = jnp.arange(10)
h[11]  # clamped to h[9]

h.at[11].set(67)  # update is dropped
# both the above behaviors can be changed with mode argument

# JAX.numpy requires passing an array or python scalar rather than silently converting smth like jnp.sum([1, 2])
# This is due to silent conversions being known to cause hidden performance problems
# make arrays explicitly
print(jnp.sum(jnp.array([1.0, 2.0, 3.0])))

# A JAX array lives on one or more devices like CPU, GPU or TPU. JAX code runs on em all.
# Typically allocates arrays to accelerators as default
# An array can even be sharded across multiple devices so JAX can run on one or thousand chips
# Every array carries a sharding attribute describing data placement

print(x.sharding)

# jax.lax can be used when numpy doesn't cut it at very niche needs.
# as a good practice stick to jnp as it is stable than JAX.lax

# ---------------------- TRANSFORMATIONS ----------------------

print("\n---------------------- TRANSFORMATIONS ----------------------")
# jax.grad() -> computes its gradient via automatic differentiation
# jav.vmap() -> from single examples to operating efficiently over batches via automatic vectorization

# The above two are direct transformation functions while the third below is just a performance transformation
# jax.jit() -> compiles the function so it runs fast

# AUTO DIFFERENTIATION
# jax.grad() takes a scalar-valued func and returns a new func that computes its gradient:

grad_tanh = jax.grad(jnp.tanh)
print(grad_tanh(2.0))

# you can stack functions and calculate grad for previous grad and thus take repeated derivatives

f = lambda x: x**3 + 2 * x**2 - 3 * x + 1

dfdx = jax.grad(f)  # 3x^2 + 4x - 3
d2fdx = jax.grad(dfdx)  # 6x + 4
d3fdx = jax.grad(d2fdx)  # 6

print(dfdx(2.0))
print(d2fdx(1.0))
print(d3fdx(1.0))

# Diff with respect to different arguments:

# jax.grad defaults to differentiation with respect to the first argument
# argnums can be used to select other arguments or several at once
# Below is a logistic regression model where we might want gradients with respect to weights and bias


def sigmoid(x):
    return (
        0.5 * (jnp.tanh(x / 2) + 1)
    )  # hyprbolic tan which simplifies to 1/(1+e^-x) which is a sigmoid. In simple terms, we squash the values into the range of 0.0 to 1.0


# Outputs probability of a label being true
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# toy dataset
inputs = jnp.array(
    [[0.52, 1.12, 0.77], [0.88, -1.08, 0.15], [0.52, 0.06, -1.30], [0.74, -2.49, 1.39]]
)

targets = jnp.array([True, True, False, True])
W = jnp.array([0.1, 0.4, -0.3])
b = 0.5


# Training loss is the neg log-likelihood of the training examples
def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))


# Differentiate loss with respect to the first positional argument
W_grad = jax.grad(
    loss, argnums=0
)(
    W, b
)  # argnums=0 is the default, so not explicitly mentioning also defaults to Differentiate with respect to the first element here(W)
print(f"{W_grad = }")

# You can choose diff vals too and drop the keyword
b_grad = jax.grad(loss, 1)(W, b)
print(f"{b_grad = }")

# Tuple values for argnums and dropping the keyword is possible too
W_grad, b_grad = jax.grad(loss, (0, 1))(W, b)
print(f"{W_grad = }")
print(f"{b_grad = }")
