from functools import partial
"""
Day 1: All six core JAX concepts in one tiny script.
Train a 2-layer MLP on y = sin(x) + noise.

Concepts you'll touch:
  1. jax.numpy as a NumPy-like API on accelerators
  2. PRNGKey + split (explicit randomness)
  3. Pytrees (params as a list of dicts)
  4. jax.grad (functional gradients)
  5. jax.jit (trace-compile a function)
  6. jax.vmap (auto-batch a single-example function)
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax.lax as lax


def init_params(key, sizes):
    params = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        key, wk, bk = random.split(key, 3)         # explicit RNG threading
        params.append({
            "W": random.normal(wk, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim),
            "b": jnp.zeros((out_dim,)),
        })
    return params


def forward(params, x):
    x = lax.cond(
        x[0] > 0, 
        lambda v: v * 2.0,
        lambda v: v * 0.5,
        x, 
    )

    for layer in params[:-1]:
      x = jnp.tanh(x @ layer["W"] + layer["b"])
      last = params[-1]
    return x @ last["W"] + last["b"]               # shape (out_dim,)


batched_forward = vmap(forward, in_axes=(None, 0))

def loss_fn(params, x_batch, y_batch):
    preds = batched_forward(params, x_batch)       # (B, 1)
    return jnp.mean((preds - y_batch) ** 2)


@jit                                     # trace-compile the step
def update(params, x_batch, y_batch, lr):
    grads = grad(loss_fn)(params, x_batch, y_batch)
    
    print(grads)
    # jax.tree.map walks both pytrees in parallel.
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

def main():
    key = random.PRNGKey(0)

    # Data: y = sin(x) on [-pi, pi], 1024 points
    key, dk = random.split(key)
    x = random.uniform(dk, (1024, 1), minval=-jnp.pi, maxval=jnp.pi)
    y = jnp.sin(x)

    # Init a (1 -> 64 -> 64 -> 1) MLP
    key, ik = random.split(key)
    params = init_params(ik, [1, 64, 64, 1])

    # Quick sanity check: gradient structure matches params structure
    g = grad(loss_fn)(params, x, y)
    print("Param tree structure: ", jax.tree.structure(params))
    print("Grad  tree structure: ", jax.tree.structure(g))
    print("Match:", jax.tree.structure(params) == jax.tree.structure(g))
    print()

    # Train
    lr = 1e-2
    for step in range(2001):
        params = update(params, x, y, lr)
        if step % 200 == 0:
            l = loss_fn(params, x, y)
            print(f"step {step:5d}  loss {l:.6f}")

if __name__ == "__main__":
    main()