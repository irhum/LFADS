import jax
import jax.numpy as jnp

def chaosrnn_df(W_y, y, gamma=2.5, tau=0.025):
    return 1/tau * (-y + gamma * W_y @ jnp.tanh(y))

def gen_chaotic_traj(key, n_traj=40, N=50, dt=0.001, total_steps=1000):
    W_y = jax.random.normal(key, shape=(N, N)) * 1/jnp.sqrt(N) 
    ys = jax.random.normal(key, shape=(n_traj, N))

    df = jax.jit(jax.vmap(lambda x: chaosrnn_df(W_y, x), in_axes=0))

    all_ys = [ys]

    for i in range(1, total_steps):
        ys = ys + dt * df(ys)
        all_ys.append(ys)

    return jnp.stack(all_ys, axis=0)

def gen_chaotic_spikes(key, n_traj=40, spikes_per_traj=10, base_rate=30, N=50, dt=0.001, seconds=1):
    total_steps = int(jnp.ceil(seconds/dt))
    trajs = gen_chaotic_traj(key, n_traj, N, dt, total_steps)

    trajs = jnp.repeat(trajs[:, :, None, :], repeats=spikes_per_traj, axis=2)
    rates = (jnp.tanh(trajs) + 1) / 2 * dt * base_rate
    spikes = jax.random.poisson(key, rates, shape=rates.shape)

    return trajs, rates, spikes