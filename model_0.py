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
def diffusion_trial(drift_g, drift_s, boundary_g, beta, D, C, SSD, boundary_s=0.25, dc=1.0, dt=.01, max_steps=2000):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.

    # go-accumulator starting point
    go_evidence = boundary_g * beta

    # is the current trial a stop trial?
    stop_trial = not(SSD == 0)

    # unlumping D into e_g, nb
    e_g = np.random.uniform(0.15, 0.25).item()
    nb = D - e_g

    # unlumping C into e_s and p_s and b
    e_s = np.random.uniform(0.15, 0.25).item()

    frac = np.random.uniform(0.25, 0.75).item()
    p_s_b = C - e_s
    p_s = p_s_b * frac
    b = p_s_b - p_s

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

    drift_g, drift_s, boundary_g, beta, D, C = params
    choicert = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i] = diffusion_trial(drift_g, drift_s, boundary_g, beta, D, C, SSD = SSDs[i])
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
    sim_data_plus = np.empty((n_sim, n_obs, 4))
    sim_data_plus[:,:,:-2] = sim_data
    sim_data_plus[:,:,2] = (sim_data_plus[:,:,1] != 0) * 1

    # add whether response was corr./incorr.
    for i in range(n_sim):
        sim_data_plus[:,:,3][i] = ((sim_data_plus[:,:,0][i] != 0) != (sim_data_plus[:,:,2][i] == 1))

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
        # 0. drift_g ~ U(-2.0, 2.0)
        # 1. drift_s ~ U(0.01, 4.0)
        # 2. boundary_g ~ U(0.1, 2.0)
        # 3. beta ~ U(0.25, 0.75)
        candidate_set = np.random.uniform(low=(-2.0, 0.01, 0.1, 0.25),
                                          high=(2.0, 4.0, 2.0, 0.75), size=(batch_size, 4))

        # 4. D = e_g + nb ~ U(0.20, 0.40)
        # 5. C = e_s + p_s + b ~ U(0.25, 0.55)

        ndts = np.random.uniform(low = (0.20, 0.25),
                                 high = (0.40, 0.55), size = (batch_size, 2))
        candidate_set = np.c_[candidate_set, ndts]

        test_data = batch_simulator(candidate_set, 100).astype(np.float32)
        n_stop_trials = sum(test_data[:,:,2][0])
        n_succ_stops = sum((test_data[:,:,0][0] == 0) & (test_data[:,:,3][0] == 1))
        perc_succ = n_succ_stops / n_stop_trials * 100

        if perc_succ > 15:
            return candidate_set.astype(np.float32)


# Variable n_trials
def prior_N(n_min=60, n_max=500):
    """
    A prior or the number of observation (will be called internally at each backprop step).
    """

    return np.random.randint(n_min, n_max + 1)


# Michael's recovery function
def recovery(possamps, truevals):  # Parameter recovery plots
    """Plots true parameters versus 99% and 95% credible intervals of recovered
    parameters. Also plotted are the median (circles) and mean (stars) of the posterior
    distributions.

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is the
    number of chains, the second to last dimension is the number of samples in
    each chain, all other dimensions must match the dimensions of truevals

    truevals : ndarray of true parameter values
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))
    truevals = np.reshape(truevals, (nvars))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    for v in range(0, nvars):
        # Compute percentiles
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Plot credible intervals
            credint = np.ones(100) * truevals[v]
            y = np.linspace(bounds[b], bounds[-1 - b], 100)
            lines = plt.plot(credint, y)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark median
                mmedian = plt.plot(truevals[v], np.median(alldata[v, :]), 'o')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(truevals[v], np.mean(alldata[v, :]), '*')
                plt.setp(mmean, markersize=10, color=teal)
    # Plot line y = x
    tempx = np.linspace(np.min(truevals), np.max(
        truevals), num=100)
    recoverline = plt.plot(tempx, tempx)
    plt.setp(recoverline, linewidth=3, color=orange)


### LOOP AND PLOT TECHNIQUE (05-30)
# set-up summary and inference networks
summary_net = InvariantNetwork()
inference_net = InvertibleNetwork({'n_params': 6})
amortizer = SingleModelAmortizer(inference_net, summary_net)
generative_model = GenerativeModel(prior, batch_simulator)
trainer = ParameterEstimationTrainer(
    network=amortizer,
    generative_model=generative_model,
    checkpoint_path="checkpoint/model_0" 
)

#data generating model
n_sim = 5000
n_trials = 1600
true_params, sim_data = generative_model(n_sim, n_trials) #n_sim, n_trials

#train 1 epoch
losses = trainer.train_experience_replay(epochs=1,
                                         batch_size=64,
                                         iterations_per_epoch=200,
                                         capacity=100,
                                         n_obs=prior_N)

# plot after 1 epoch
random.seed(1999)
np.random.seed(1999)
tf.random.set_seed(1999)

n_plot_points = 5
plot_priors = prior(n_plot_points)
plot_data = batch_simulator(plot_priors, n_trials).astype(np.float32)

plot_param_samples = amortizer.sample(plot_data, n_samples=1600)
plot_param_means = plot_param_samples.mean(axis=0)
filename = "img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/true_vs_est_epoch_1"
true_vs_estimated(plot_priors,
                  plot_param_means,
                  ['drift_g', 'drift_s', 'boundary_g', 'beta', 'D', 'C'],
                  filename = filename,
                  figsize = (28,5),
                  dpi = 600,
                  show = False)

# drift_go
plt.figure()
recovery(plot_param_samples[:, :, 0].T.reshape(n_plot_points, n_trials, 1), plot_priors[:, 0].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('drift_go')
drift_go_filename = 'img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/drift_go_epoch_1'
plt.savefig(drift_go_filename, dpi = 300)

# drift_stop
plt.figure()
recovery(plot_param_samples[:, :, 1].T.reshape(n_plot_points, n_trials, 1), plot_priors[:, 1].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('drift_stop')
drift_stop_filename = 'img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/drift_stop_epoch_1'
plt.savefig(drift_stop_filename, dpi = 300)

# boundary_go
plt.figure()
recovery(plot_param_samples[:, :, 2].T.reshape(n_plot_points, n_trials, 1), plot_priors[:, 2].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('boundary_go')
boundary_go_filename = 'img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/boundary_go_epoch_1'
plt.savefig(boundary_go_filename, dpi = 300)

#plot parameter estimates every 10 epochs
for i in range(0, 100):
    start_time = perf_counter()

    losses = trainer.train_experience_replay(epochs=10,
                                             batch_size=64,
                                             iterations_per_epoch=200,
                                             capacity=100,
                                             n_obs=prior_N) 

    plot_param_samples = amortizer.sample(plot_data, n_samples=1600)
    plot_param_means = plot_param_samples.mean(axis=0)
    filename = "img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/true_vs_est_epoch_%s" % str((i+1)*10 + 1)
    true_vs_estimated(plot_priors,
                      plot_param_means,
                      ['drift_g', 'drift_s', 'boundary_g', 'beta', 'D', 'C'],
                      filename = filename,
                      figsize = (28,5),
                      dpi = 600,
                      show = False)

    # drift_go
    plt.figure()
    recovery(plot_param_samples[:, :, 0].T.reshape(n_plot_points, n_trials, 1), plot_priors[:, 0].squeeze())
    plt.xlabel('True')
    plt.ylabel('Posterior')
    plt.title('drift_go')
    drift_go_filename = 'img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/drift_go_epoch_%s' % str((i+1)*10 + 1)
    plt.savefig(drift_go_filename, dpi = 300)

    # drift_stop
    plt.figure()
    recovery(plot_param_samples[:, :, 1].T.reshape(n_plot_points, n_trials, 1), plot_priors[:, 1].squeeze())
    plt.xlabel('True')
    plt.ylabel('Posterior')
    plt.title('drift_stop')
    drift_stop_filename = 'img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/drift_stop_epoch_%s' % str((i+1)*10 + 1)
    plt.savefig(drift_stop_filename, dpi = 300)

    # boundary_go
    plt.figure()
    recovery(plot_param_samples[:, :, 2].T.reshape(n_plot_points, n_trials, 1), plot_priors[:, 2].squeeze())
    plt.xlabel('True')
    plt.ylabel('Posterior')
    plt.title('boundary_go')
    boundary_go_filename = 'img/DDM_fix_a_s_low_d_g_cut_no_stop_D_C/boundary_go_epoch_%s' % str((i+1)*10 + 1)
    plt.savefig(boundary_go_filename, dpi = 300)

    #clear all figures
    plt.close('all')
    plt.clf()
    plt.cla()

    end_time = perf_counter()
    duration = end_time - start_time
    print(duration)