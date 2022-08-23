import os
import numpy as np
import pandas as pd
from scipy import stats
from time import time, perf_counter
import matplotlib.pyplot as plt
import random

from numba import njit
import tensorflow as tf

from bayesflow.networks import InvertibleNetwork, InvariantNetwork
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.diagnostics import *
from bayesflow.models import GenerativeModel


@njit
def diffusion_trial(drift_g, drift_s, boundary_g, boundary_s, beta, e_g, nb, b, e_s, p_s, SSD, dc=1.0, dt=.01, max_steps=2000):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.

    # go-accumulator starting point
    go_evidence = boundary_g * beta

    # is the current trial a stop trial?
    stop_trial = not(SSD == 0)

    # GO trial
    if stop_trial == False:
        while (go_evidence > 0 and go_evidence < boundary_g and n_steps < max_steps):

            # increment step
            n_steps += 1.0

            # "wait" until e_g (VET_go) has elapsed
            if (n_steps - e_g/dt) < 0:
                continue

            # DDM equation
            go_evidence += drift_g*dt + np.sqrt(dt) * dc * np.random.normal()

        t = n_steps * dt
        if go_evidence >= boundary_g:
            choicert = t + nb + b
        elif go_evidence <= 0:
            choicert = -t - nb - b
        else:
            choicert = 0
        return choicert

    # STOP trial
    else:
        stop_evidence = 0.
        while (go_evidence > 0 and go_evidence < boundary_g and stop_evidence < boundary_s and n_steps < max_steps):

            n_steps += 1.0

            # "wait" until e_g (VET_go) has elapsed
            if (n_steps - e_g/dt) < 0:
                continue

            go_evidence += drift_g*dt + np.sqrt(dt) * dc * np.random.normal()

            # "wait" until SSD and e_s (VET_stop) have elapsed
            if (n_steps - SSD/dt - e_s/dt) <= 0:
                continue

            stop_evidence += drift_s*dt + np.sqrt(dt) * dc * np.random.normal()

        # stop-accumulator boundary reached
        if stop_evidence >= boundary_s:

            # time between stimulus presentation and stop-accumulator finishing
            t_s = n_steps*dt

            # stop production time (in timestep size)
            p_s_steps = round(p_s/dt)
            for i in range(p_s_steps):

                n_steps += 1.0
                go_evidence += drift_g*dt + np.sqrt(dt) * dc * np.random.normal()

                # go-accumulator boundary reached
                if go_evidence >= boundary_g or go_evidence <= 0:

                    # time between stimulus presentation and go-accumulator finishing
                    t_g = n_steps*dt

                    # has the ballistic component begun?
                    if t_g + nb > t_s + p_s:
                        return 0
                    else:
                        return np.sign(go_evidence) * (t_g + nb + b)

            # stop-production time elapsed & go-accumulator boundary not reached
            return 0

        # go-accumulator boundary reached
        elif go_evidence >= boundary_g or go_evidence <= 0:

            t_g = n_steps*dt

            # non-ballistic component duration (in timestep size)
            nb_steps = round(nb/dt)
            for i in range(nb_steps):

                n_steps += 1.0
                stop_evidence += drift_s*dt + np.sqrt(dt) * dc * np.random.normal()

                # stop-accumulator boundary reached
                if stop_evidence >= boundary_s:

                    t_s = n_steps*dt

                    if t_g + nb < t_s + p_s:
                        return np.sign(go_evidence) * (t_g + nb + b)
                    else:
                        return 0

            # non-ballistic component elapsed & stop-accumulator boundary not reached
            return np.sign(go_evidence) * (t_g + nb + b)

        # max_steps reached, no decision
        else:
            return 0


@njit
def diffusion_condition(params, n_trials, SSDs):
    """Simulates a diffusion process over an entire condition."""

    drift_g, drift_s, boundary_g, boundary_s, beta, e_g, nb, b, e_s, p_s = params
    choicert = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i] = diffusion_trial(drift_g, drift_s, boundary_g, boundary_s, beta, e_g, nb, b, e_s, p_s, SSD = SSDs[i])
    return choicert


def batch_simulator(prior_samples, n_obs, dt=0.005, s=1.0, max_iter=1e4):
    """
    Simulate multiple diffusion_model_datasets.
    """

    n_sim = prior_samples.shape[0] # number of sets of parameters
    sim_data = np.empty((n_sim, n_obs), dtype=np.float32)
    stop_arr = np.empty((n_sim, n_obs), dtype=np.float32)
    theta_stop = np.random.uniform(0.2, 0.4, n_sim)

    # Simulate diffusion data
    for i in range(n_sim):

        # Create SSDs
        SSDs = np.random.binomial(1, theta_stop[i], n_obs).astype(np.float32)
        for j in range(len(SSDs)):
            if SSDs[j] == 1:
                SSDs[j] = np.random.uniform(0.1,0.3)

        stop_arr[i] = SSDs
        sim_data[i] = diffusion_condition(prior_samples[i], n_obs, SSDs)

    # let's make this 3D, add SSDs
    sim_data = np.stack((sim_data, stop_arr), axis=-1)

    # add trial type
    sim_data_plus = np.empty((n_sim, n_obs, 6))
    sim_data_plus[:,:,:-4] = sim_data
    sim_data_plus[:,:,2] = (sim_data_plus[:,:,1] != 0) * 1

    # add whether response was corr./incorr.
    for i in range(n_sim):
        sim_data_plus[:,:,3][i] = ((sim_data_plus[:,:,0][i] != 0) != (sim_data_plus[:,:,2][i] == 1))

    # add VETs (e_g, e_s)
    for i in range(n_sim):
        sim_data_plus[:,:,4][i] = prior_samples[i, 5] # e_g
        sim_data_plus[:,:,5][i] = prior_samples[i, 8] # e_s

    return sim_data_plus


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------

    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    ----------

    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
    """
    while True:
        # Prior ranges for the simulator:
        # 0. drift_g ~ U(-7.0, 7.0)
        # 1. drift_s ~ U(0.01, 7.0)
        # 2. boundary_g ~ U(0.1, 2.0)
        # 3. boundary_s ~ U(0.1, 2.0)
        # 4. beta ~ U(0.20, 0.80)
        p_samples = np.random.uniform(low=(-2.0, 0.01, 0.1, 0.1, 0.25),
                                      high=(2.0, 4.0, 2.0, 1.0, 0.75), size=(batch_size, 5))

        # ndt_go = e_g + nb + b
        # 5. e_g ~ U(0.15, 0.25) //i.e., VET_go
        # 6. nb ~ U(0.01, 0.2)
        # 7. b ~ U(0.01, 0.2)
        ndt_go_comp = np.random.uniform(low = (0.15, 0.01, 0.01),
                                        high = (0.25, 0.2, 0.2), size = (batch_size, 3))
        p_samples = np.c_[p_samples, ndt_go_comp]

        # ndt_stop = e_s + p_s
        # 8. e_s ~ U(0.15, 0.25) //i.e., VET_stop
        # 9. p_s ~ U(0.01, 0.1)
        ndt_stop_comp = np.random.uniform(low = (0.15, 0.01),
                                          high = (0.25, 0.1), size = (batch_size, 2))

        candidate_set = np.c_[p_samples, ndt_stop_comp]
        test_data = batch_simulator(candidate_set, 100).astype(np.float32)
        n_stop_trials = sum(test_data[:,:,2][0])
        n_succ_stops = sum((test_data[:,:,0][0] == 0) & (test_data[:,:,3][0] == 1))
        perc_succ = n_succ_stops / n_stop_trials * 100

        if perc_succ > 15:
            return candidate_set.astype(np.float32)


# check distribution of successful stopping
perc_succ = np.empty(10000)
for i in range(0,10000):
    true_params = prior(1)
    x = batch_simulator(true_params, 800).astype(np.float32)

    n_stop_trials = sum(x[:,:,2][0])
    n_succ_stops = sum((x[:,:,0][0] == 0) & (x[:,:,3][0] == 1))
    perc_succ[i] = n_succ_stops / n_stop_trials * 100

plt.figure()
plt.hist(perc_succ, bins = 100)
plt.show()


# Variable n_trials
def prior_N(n_min=60, n_max=300):
    """
    A prior or the number of observation (will be called internally at each backprop step).
    """

    return np.random.randint(n_min, n_max + 1)


#Test input & output
# timethen = time()
# print(batch_simulator(prior(50),100).shape)
# print('Test simulator took %.3f secs' % (time()-timethen))
summary_net = InvariantNetwork()
inference_net = InvertibleNetwork({'n_params': 10})
amortizer = SingleModelAmortizer(inference_net, summary_net)
generative_model = GenerativeModel(prior, batch_simulator)
trainer = ParameterEstimationTrainer(
    network=amortizer,
    generative_model=generative_model,
    checkpoint_path="checkpoint/model_1" 
)

# Pre-simulated data (could be loaded from somewhere else)
n_sim = 5000
n_trials = 1600
true_params, sim_data = generative_model(n_sim, n_trials)

start_time = perf_counter() 

## Round-based training
# losses = trainer.train_rounds(epochs=1, rounds=5, sim_per_round=200, batch_size=32, n_obs=n_trials)

# Experience-replay training
losses = trainer.train_experience_replay(epochs=1000,
                                         batch_size=64,
                                         iterations_per_epoch=200,
                                         capacity=100,
                                         n_obs=prior_N)

end_time = perf_counter()
duration = end_time - start_time
print(duration)

# Validate (quick and dirty)
n_param_sets = 300
n_samples = 1600
true_params = prior(n_param_sets)
x = batch_simulator(true_params, n_samples).astype(np.float32)
param_samples = amortizer.sample(x, n_samples=n_samples)
param_means = param_samples.mean(axis=0)
true_vs_estimated(true_params[(np.all(param_means[:, list(range(5,10))] > 0.01, axis = 1)), :],
                  param_means[(np.all(param_means[:, list(range(5,10))] > 0.01, axis = 1)), :],
                  ['drift_g', 'drift_s', 'boundary_g', 'boundary_s', 'beta', 'e_g', 'nb', 'b', 'e_s', 'p_s'],
                  filename = "NDDM_full_cut_no_stop")

columns = ['delta_g', 'delta_s', 'alpha_g', 'alpha_s', 'beta', 'e_g', 'nb', 'b', 'e_s', 'p_s']
def plot_est1_vs_est2(par_array, par1, par2):
    df = pd.DataFrame(par_array,
                      columns = columns)

    m, b = np.polyfit(df[par1], df[par2], 1)
    r_sq = np.corrcoef(df[par1], df[par2])[0,1]**2
    r_sq_box = 'R^2 = %.2f' % r_sq

    plt.scatter(df[par1], df[par2], color = 'black', alpha = 0.5)
    plt.plot(df[par1], m*df[par1]+b, color = 'red')
    plt.text(df[par1].min(), df[par2].min(), r_sq_box, fontsize = 12)
    plt.xlabel(par1)
    plt.ylabel(par2)
    plt.show()

# plot_est1_vs_est2(param_means, 'drift_s', 'drift_g')
# plot_est1_vs_est2(param_means, 'drift_s', 'boundary_g')
# plot_est1_vs_est2(param_means, 'delta_s', 'alpha_s')
# plot_est1_vs_est2(param_means, 'drift_s', 'beta')
# plot_est1_vs_est2(param_means, 'drift_s', 'C')

# plot_est1_vs_est2(param_means, 'boundary_s', 'drift_g')
# plot_est1_vs_est2(param_means, 'boundary_s', 'boundary_g')
# plot_est1_vs_est2(param_means, 'boundary_s', 'beta')
# plot_est1_vs_est2(param_means, 'boundary_s', 'C')

param_means_df = pd.DataFrame(param_means, columns = columns)

plt.figure(figsize=(12, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(param_means_df.corr(), dtype=np.bool))
heatmap = sns.heatmap(param_means_df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG', fmt = '.2f')
heatmap.set_title('Posterior Parameter Estimate Correlation Matrix', fontdict={'fontsize':18}, pad=16);
plt.show()