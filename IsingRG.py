# %%
"""Importing the libraries"""
import jax
import jax.numpy as jnp
from flax.linen import avg_pool
from functools import partial
from jax.scipy.signal import convolve2d


def ComputeLocalField(configuration, pos, d):
    """
    Takes a configuration and a position and returns the sum
    of the sites at distance d
    """
    x0, y0 = pos
    L = configuration.shape[-1]
    h = 0
    for cursor in range(1, d):
        h += configuration[(x0 + cursor) % L, (y0 + d - cursor + L) % L]
        h += configuration[(x0 - cursor) % L, (y0 + d - cursor + L) % L]
        h += configuration[(x0 + cursor) % L, (y0 - d + cursor + L) % L]
        h += configuration[(x0 - cursor) % L, (y0 - d + cursor + L) % L]
    h += configuration[(x0 + d) % L, y0]
    h += configuration[(x0 - d + L) % L, y0]
    h += configuration[x0, (y0 + d) % L]
    h += configuration[x0, (y0 - d + L) % L]
    return h


@jax.jit
def Make_MontecarloStep(configuration, rng, K):
    """
    Take  a configuration of an ising model and evolves it using the couplings K and
    using the rng key rng. The function returns the final configuration and its magnetization
    """
    L = configuration.shape[-1]
    rng_pos, rng_p_acc = jax.random.split(rng, 2)
    pos = tuple(jax.random.randint(rng_pos, (2,), minval=0, maxval=L))
    p = jax.random.uniform(rng_p_acc)
    H = jnp.zeros((1,))
    for distance in range(len(K)):
        H += K[distance] * ComputeLocalField(configuration, pos, distance + 1)
    flip_probability = jnp.exp(-2 * H * configuration[pos])
    configuration = jnp.where(p < flip_probability, configuration.at[pos].multiply(-1), configuration)
    return configuration, configuration.sum() / (L * L)


def Make_MonteCarloStepS(configuration, rng, num_flips, K):
    """
    evolve the ising configuration for num_flips montecarlo steps using th
    couplings K and random key rng
    """
    rng_new, rng = jax.random.split(rng, 2)
    rng_s = jax.random.split(rng_new, num_flips)
    configuration, rng = jax.lax.scan(partial(Make_MontecarloStep, K=K), configuration, rng_s)
    return configuration, configuration


def Sample(initial_configuration, rng, num_flips, num_samples, K):
    """
    This function simulate an ising model with couplings K= [K_d=1, K_d=2, ...], starting from the
     configuration initial_configuration and return num_samples configurations
    sampled every num_flips individual (single spin) montecarlo steps.
    """
    rng_sampling = jax.random.split(rng, num_samples)
    final_configuration, sampled_configurations = jax.lax.scan(partial(Make_MonteCarloStepS, K=K, num_flips=num_flips),
                                                               initial_configuration, rng_sampling)
    return sampled_configurations


@jax.jit
def Apply_Majority_Rule(x, rng):
    """
    assign 1,-1 depending on the sign of the argument.
    It returns a random result (using the key rng) if the argument is exactly zero.
    """
    return jnp.where(x > 0, 1,
                     jnp.where(x == 0, jax.random.randint(rng, (1,), minval=0, maxval=2) * 2 - 1, -1))


def Block_configuration(s, block_size, rng):
    """
    The function arguments are s = (Num_samples, L, L), block_size and a RNG seed.
    It returns the corresponding blocked configurations in the format
    (Num_samples, L/block_size, L/block_size.
    The size of the system must be a multiple of the block size.
    """
    L = s.shape[-1]
    if s.shape[-2] != L:
        print("The system must be square size")
        return
    if L % block_size != 0:
        print("The block size is not compatible with the system size")
        return
    Block_shape = (block_size, block_size)
    """The built in function avg_pool is equivalent to, 
            [[ b[1,   k_x * b_s:(k_x + 1) * b_s, k_y * b_s:(k_y + 1)*b_s].sum() for k_y in
                range(Lb) ] for k_x in range(Lb)] 
                but it is way more efficient """
    blocked = avg_pool(s.transpose(1, 2, 0), window_shape=Block_shape, strides=Block_shape).transpose(2, 0, 1)

    """Now we flatten the output and use the function vmap to each of the elements. 
    This is just an efficient way to apply the function Apply_Majority_Rule to each of the element 
    of blocked"""

    """Maybe the pooling can be realized also using a 2d convolution and we do not need flax.linens"""
    B, L1, L2 = blocked.shape
    blocked_flat = blocked.reshape(B * L1 * L2)
    rng_s = jax.random.split(rng, B * L1 * L2)
    blocked_flat_majority = jax.vmap(Apply_Majority_Rule, in_axes=(0, 0), out_axes=(0))(blocked_flat, rng_s)
    blocked_majority = blocked_flat_majority.reshape((B, L1, L2))
    return blocked_majority


# @jax.jit
def PadConfiguration(s, padding):
    upper_pad = s[:, -padding:]
    lower_pad = s[:, :padding]
    partial_lattice = jnp.concatenate([upper_pad, s, lower_pad], axis=1)
    left_pad = partial_lattice[-padding:, :]
    right_pad = partial_lattice[:padding, :]
    padded_lattice = jnp.concatenate([left_pad, partial_lattice, right_pad], axis=0)
    return padded_lattice


@jax.jit
def build_filter(K):
    """
    :jnp.array K: list of couplings.
    :return: Build a convolutional filter that corresponds to the
    couplings K.
     the size of the filter is 2*len(K)+1. For example if the couplings K
    are of the form [k1,k2] the function returns:
    [[0   0   k2  0   0],
     [0   k2  k1  k2  0],
     [k2  k1  0   k1  k2],
     [0   k2  k1  k2   0],
     [0   0   k2  0   0]]
    """
    D_max = len(K)
    Filter = jnp.zeros((len(K) * 2 + 1, len(K) * 2 + 1))
    Lf, Lf = Filter.shape
    for i in range(Lf):
        for j in range(Lf):
            distance = abs(i - D_max) + abs(j - D_max)
            if 0 < distance <= len(K):
                Filter = Filter.at[i, j].set(K[distance - 1])
    return Filter


@jax.jit
def ComputeLocalFields(padded_lattice, filter):
    """ Compute all the local field using the couplings encoded in the filter. For example
    if the couplings are of the form [k1,k2] the filter must have the form:
    [[0   0   k2  0   0],
     [0   k2  k1  k2  0],
     [k2  k1  0   k1  k2],
     [0   k2  k1  k2   0],
     [0   0   k2  0   0]]
     The input of the function is the configuration padded using period boundary condition, where the padding
     corresponds to the maximum coupling distance.
    """
    return convolve2d(padded_lattice, filter, mode='valid')


# %%

@jax.jit
def Pseudo_Loss(K, St):
    """
    :jnp.array K:
    :jnp.array St: ising model configurations  size [num_samples, L,L]
    :return: the pseudo loss corresponding to the pseudo-likelihood (with a minus in front) that
    a model with couplings K generated the dataset St
    """
    Filter = build_filter(K)
    padded_configurations = jax.vmap(PadConfiguration, in_axes=(0, None))(St, len(K))
    H_n_t = jax.vmap(ComputeLocalFields, in_axes=(0, None), out_axes=0)(padded_configurations, Filter)
    normalization = 1. / (1 + jnp.exp(-2 * jnp.multiply(H_n_t, St)))
    ln = jnp.average(jnp.log(normalization), axis=1)
    return -jnp.average(ln)


@jax.jit
def Pseudo_Loss_fn_and_grad(K, St):
    """
    :jnp.array K:
    :jnp.array St: ising model configurations  size [num_samples, L,L]
    :return: the pseudo loss and its gradient of the loss corresponding to the pseudo-likelihood (with a minus in front) that
    a model with couplings K generated the dataset St.  The gradient is computed with respect to the couplings K
    """
    Filter = build_filter(K)
    padded_configurations = jax.vmap(PadConfiguration, in_axes=(0, None))(St, len(K))
    H_n_t = jax.vmap(ComputeLocalFields, in_axes=(0, None), out_axes=0)(padded_configurations, Filter)
    normalization_1 = 1. / (1 + jnp.exp(-2 * jnp.multiply(H_n_t, St)))
    ln = jnp.average(jnp.log(normalization_1), axis=1)
    grad = []
    normalization_2 = 1. / (1 + jnp.exp(2 * jnp.multiply(H_n_t, St)))
    temp = jnp.multiply(normalization_2, St)
    for d in range(len(K)):
        _uniform_couplings = [0 for i in range(d + 1)]
        _uniform_couplings[-1] = 1
        Flat_filter_d = build_filter(_uniform_couplings)
        padded_configurations = jax.vmap(PadConfiguration, in_axes=(0, None))(St, d + 1)
        Avg_d = jax.vmap(ComputeLocalFields, in_axes=(0, None), out_axes=0)(padded_configurations, Flat_filter_d)
        grad_d = -2 * jnp.average(jnp.multiply(Avg_d, temp))
        grad.append(grad_d)
    return jnp.average(jnp.array(-ln)), jnp.array(grad)
