# TODO: fix, is exp really the way to cause firing?
# real neurons probably have saturation points, not exponential firing?

import jax
import jax.numpy as jnp

DIM = 3
rng_key = jax.random.PRNGKey(40)

def lorenz_df(x, sigma=10.0, rho=28.0, beta=8/3):
    dx0 = sigma * (x[1] - x[0])
    dx1 = x[0] * (rho - x[2]) - x[1]
    dx2 = x[0] * x[1] - beta * x[2]

    return jnp.array([dx0, dx1, dx2])

def get_inital_states(n_inits):
    init_conds = jax.random.normal(rng_key, shape=(n_inits, DIM)) 
    return init_conds + jnp.array([[0, 0, 25]])

def burn_inits(init_conds, burn_steps=500, step_size=0.005):
    f = jax.jit(jax.vmap(lorenz_df, in_axes=0))

    for _ in range(burn_steps):
        init_conds += step_size * f(init_conds)

    return init_conds

# T, N_init, D
def generate_trajectories(init_conds, step_size=0.005, n_steps=1000, burn_in=True, normalize=True):
    if burn_in: init_conds = burn_inits(init_conds, step_size=step_size)
     
    f = jax.jit(jax.vmap(lorenz_df, in_axes=0))

    states = [init_conds]

    curr_conds = init_conds

    for _ in range(1, n_steps):
        curr_conds += step_size * f(curr_conds)
        states.append(curr_conds)

    states = jnp.stack(states, axis=0)

    if normalize:
        states = states - jnp.mean(states, axis=(0,1))
        states = states / jnp.max(jnp.abs(states), axis=(0,1))
        
    return states

def _history_step(key, trajs_t, W, b, history, n_trials, has_lag=True):
    w_traj_t = trajs_t @ W
    w_traj_rep = jnp.repeat(w_traj_t[:, None, :], repeats=n_trials, axis=1)

    rates_t = jnp.exp(jnp.minimum(w_traj_rep + jnp.einsum('ijkl, i -> jkl', history, b), 10))
    spikes_t = jax.random.poisson(key, rates_t, shape=rates_t.shape).clip(0,1)

    if has_lag:
        history = history.at[2:-1, :, :].set(history[1:-2, :, :])
        history = history.at[1, :, :, :].set(spikes_t)
    
    return history, rates_t, spikes_t

def generate_history_weighting(key, trajs, W, b, n_trials=1):
    # T, N_init, D
    t_steps = trajs.shape[0]
    n_lag_window = b.shape[0]
    n_neurons = W.shape[1]
    n_trajs = trajs.shape[1]

    rates_per_t = []
    spikes_per_t = []

    history = jnp.zeros((n_lag_window, n_trajs, n_trials, n_neurons))
    history = history.at[0, :, :, :].set(1)

    for i in range(t_steps):
        history, rates_t, spikes_t = _history_step(key, trajs[i, :, :], W, b, history, 
                                                    n_trials=n_trials, has_lag=n_lag_window > 1)
        
        rates_per_t.append(rates_t)
        spikes_per_t.append(spikes_t)

    return jnp.stack(rates_per_t, axis=0), jnp.stack(spikes_per_t, axis=0)

def generate_spike_rates(n_trajs, n_spikes_per_traj, n_steps=1000, n_channels=30, base_rate=5):
    inits = get_inital_states(n_trajs)
    trajs = generate_trajectories(inits, n_steps=n_steps)

    W = base_rate * (jax.random.uniform(rng_key, shape=(3, n_channels)) + 1)
    W = W * jnp.sign(jax.random.normal(rng_key, shape=(3, n_channels)))

    b = jnp.array([jnp.log(base_rate/n_steps),-10,-10,-3,-3,-3,-3,-2,-2,-1,-1])
    
    return trajs, generate_history_weighting(rng_key, trajs, W, b, n_trials=n_spikes_per_traj)
