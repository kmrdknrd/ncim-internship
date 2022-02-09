# Online references:
# (Accessed 2021-Sep-01):
# https://bayesflow.readthedocs.io/en/latest/tutorial_notebooks/Parameter_Estimation_Workflow.html
# Solve iProgress not found error:
# https://stackoverflow.com/questions/67998191/importerror-iprogress-not-found-please-update-jupyter-and-ipywidgets-although

# Notes:
# I had to use a Python 3.8 virtual environment that was outside of Anaconda Python in order to install tensorflow
# Neural networks are saved after each trainer run. So a model can be closer to the truth with different trainers.

# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 30-Sep-21        Michael Nunez                  Converted from Parameter_Estimation_Workflow.py


import os
import numpy as np
from scipy import stats
from time import time
import matplotlib.pyplot as plt

from numba import njit
import tensorflow as tf

from bayesflow.networks import InvertibleNetwork, InvariantNetwork
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.diagnostics import *
from bayesflow.models import GenerativeModel


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

    # Prior ranges for the simulator
    # drift ~ U(-7.0, 7.0)
    # boundary ~ U(0.1, 4.0)
    # ndt ~ U(0.1, 3.0)
    # beta ~ U(0.0, 1.0)  # relative start point
    # dc ~ U(0.1, 4.0)   # Diffusion coefficient
    n_parameters = 5
    p_samples = np.random.uniform(low=(-7.0, 0.1, 0.1, 0.0, 0.1),
                                  high=(7.0, 3.0, 4.0, 1.0, 4.0), size=(batch_size, n_parameters))
    return p_samples.astype(np.float32)


@njit
def diffusion_trial(drift, boundary, ndt, beta=0.5, dc=1.0, dt=.01, max_steps=2000):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    evidence = boundary * beta

    # Simulate a single DM path
    while (evidence > 0 and evidence < boundary and n_steps < max_steps):

        # DDM equation
        evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt
    if evidence >= boundary:
        choicert = rt + ndt
    elif evidence <= 0:
        choicert = -rt - ndt
    else:
        choicert = np.sign(evidence - boundary*.5)*(rt + ndt)  # Choose closest boundary at max_steps
    return choicert


# Simulate diffusion models
@njit
def simulratcliff(Nu=1, Alpha=1, Tau=.4, Beta=.5,  Varsigma=1, rangeTau=0, rangeBeta=0, Eta=0):
    """
    SIMULRATCLIFF  Generates data according to a drift diffusion model with optional trial-to-trial variability


    Reference:
    Tuerlinckx, F., Maris, E.,
    Ratcliff, R., & De Boeck, P. (2001). A comparison of four methods for
    simulating the diffusion process. Behavior Research Methods,
    Instruments, & Computers, 33, 443-456.

    Parameters
    ----------
    N: a integer denoting the size of the output vector
    (defaults to 100 experimental trials)

    Alpha: the mean boundary separation across trials  in evidence units
    (defaults to 1 evidence unit)

    Tau: the mean non-decision time across trials in seconds
    (defaults to .4 seconds)

    Nu: the mean drift rate across trials in evidence units per second
    (defaults to 1 evidence units per second, restricted to -5 to 5 units)

    Beta: the initial bias in the evidence process for choice A as a proportion of boundary Alpha
    (defaults to .5 or 50% of total evidence units given by Alpha)

    rangeTau: Non-decision time across trials is generated from a uniform
    distribution of Tau - rangeTau/2 to  Tau + rangeTau/2 across trials
    (defaults to 0 seconds)

    rangeZeta: Bias across trials is generated from a uniform distribution
    of Zeta - rangeZeta/2 to Zeta + rangeZeta/2 across trials
    (defaults to 0 evidence units)

    Eta: Standard deviation of the drift rate across trials
    (defaults to 3 evidence units per second, restricted to less than 3 evidence units)

    Varsigma: The diffusion coefficient, the standard deviation of the
    evidence accumulation process within one trial. It is recommended that
    this parameter be kept fixed unless you have reason to explore this parameter
    (defaults to 1 evidence unit per second)

    Returns
    -------
    Numpy complex vector with  reaction times (in seconds) multiplied by the response vector
    such that negative reaction times encode response B and positive reaction times
    encode response A

    Also returns single-trial drift rates


    Converted from simuldiff.m MATLAB script by Joachim Vandekerckhove
    See also http://ppw.kuleuven.be/okp/dmatoolbox.
    """

    if (Nu < -5) or (Nu > 5):
        Nu = np.sign(Nu) * 5
        # warnings.warn('Nu is not in the range [-5 5], bounding drift rate to %.1f...' % (Nu))

    if (Eta > 3):
        # warning.warn('Standard deviation of drift rate is out of bounds, bounding drift rate to 3')
        eta = 3

    if (Eta == 0):
        Eta = 1e-16

    # Called sigma in 2001 paper
    D = np.power(Varsigma, 2) / 2

    # Program specifications
    eps = 2.220446049250313e-16  # precision from 1.0 to next double-precision number
    delta = eps

    r1 = np.random.normal()
    mu = Nu + r1 * Eta
    bb = Beta - rangeBeta / 2 + rangeBeta * np.random.uniform(0,1)  #Numba likes np.random.uniform(0,1) not ()
    zz = bb * Alpha
    finish = 0
    totaltime = 0
    startpos = 0
    Aupper = Alpha - zz
    Alower = -zz
    radius = np.min(np.array([np.abs(Aupper), np.abs(Alower)]))
    while (finish == 0):
        lambda_ = 0.25 * np.power(mu, 2) / D + 0.25 * D * np.power(np.pi, 2) / np.power(radius, 2)
        # eq. formula (13) in 2001 paper with D = sigma^2/2 and radius = Alpha/2
        F = D * np.pi / (radius * mu)
        F = np.power(F, 2) / (1 + np.power(F, 2))
        # formula p447 in 2001 paper
        prob = np.exp(radius * mu / D)
        prob = prob / (1 + prob)
        dir_ = 2 * (np.random.uniform(0,1) < prob) - 1
        l = -1
        s2 = 0
        while (s2 > l):
            s2 = np.random.uniform(0,1)
            s1 = np.random.uniform(0,1)
            tnew = 0
            told = 0
            uu = 0
            while (np.abs(tnew - told) > eps) or (uu == 0):
                told = tnew
                uu = uu + 1
                tnew = told + (2 * uu + 1) * np.power(-1, uu) * np.power(s1, (F * np.power(2 * uu + 1, 2)))
                # infinite sum in formula (16) in BRMIC,2001
            l = 1 + np.power(s1, (-F)) * tnew
        # rest of formula (16)
        t = np.abs(np.log(s1)) / lambda_
        # is the negative of t* in (14) in BRMIC,2001
        totaltime = totaltime + t
        dir_ = startpos + dir_ * radius
        ndt = Tau - rangeTau / 2 + rangeTau * np.random.uniform(0,1)
        if ((dir_ + delta) > Aupper):
            T = ndt + totaltime
            XX = 1
            finish = 1
        elif ((dir_ - delta) < Alower):
            T = ndt + totaltime
            XX = -1
            finish = 1
        else:
            startpos = dir_
            radius = np.min(np.abs(np.array([Aupper, Alower]) - startpos))

    result = T * XX
    return result


@njit
def diffusion_condition(params, n_trials):
    """Simulates a diffusion process over an entire condition."""

    drift, boundary, ndt, beta, dc = params
    choicert = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i] = diffusion_trial(drift, boundary, ndt, beta, dc)
        # choicert[i] = simulratcliff(drift, boundary, ndt, beta, dc)  # Numba likes this line a certain way
    return choicert


def batch_simulator(prior_samples, n_obs, dt=0.005, s=1.0, max_iter=1e4):
    """
    Simulate multiple diffusion_model_datasets.
    """

    n_sim = prior_samples.shape[0]
    sim_data = np.empty((n_sim, n_obs), dtype=np.float32)

    # Simulate diffusion data
    for i in range(n_sim):
        sim_data[i] = diffusion_condition(prior_samples[i], n_obs)

    # For some reason BayesFlow wants there to be at least two data dimensions
    sim_data = sim_data.reshape(n_sim, n_obs, 1)
    return sim_data


#Test input & output
timethen = time()
print(batch_simulator(prior(50),100).shape)
print('Test simulator took %.3f secs' % (time()-timethen))
summary_net = InvariantNetwork()
inference_net = InvertibleNetwork({'n_params': 5})
amortizer = SingleModelAmortizer(inference_net, summary_net)

generative_model = GenerativeModel(prior, batch_simulator)

trainer = ParameterEstimationTrainer(
    network=amortizer,
    generative_model=generative_model,
)

# Pre-simulated data (could be loaded from somewhere else)
n_sim = 5000
n_trials = 200
true_params, sim_data = generative_model(n_sim, n_trials)

# Using internally simulated data
losses = trainer.train_offline(epochs=1, batch_size=64, params=true_params, sim_data=sim_data)

# Online training
losses = trainer.simulate_and_train_offline(n_sim=1000, epochs=2, batch_size=32, n_obs=n_trials)


# Variable n_trials
def prior_N(n_min=60, n_max=300):
    """
    A prior or the number of observation (will be called internally at each backprop step).
    """

    return np.random.randint(n_min, n_max + 1)


losses = trainer.train_online(epochs=2, iterations_per_epoch=100, batch_size=32, n_obs=prior_N)

# Round-based training
losses = trainer.train_rounds(epochs=1, rounds=5, sim_per_round=200, batch_size=32, n_obs=n_trials)

# Experience-replay training
losses = trainer.train_experience_replay(epochs=300,
                                         batch_size=32,
                                         iterations_per_epoch=100,
                                         capacity=100,
                                         n_obs=prior_N)

# Validate (quick and dirty)
n_param_sets = 300
n_samples = 1000
true_params = prior(n_param_sets)
x = batch_simulator(true_params, n_samples).astype(np.float32)
param_samples = amortizer.sample(x, n_samples=n_samples)
param_means = param_samples.mean(axis=0)
true_vs_estimated(true_params, param_means, ['drift', 'boundary', 'ndt', 'beta', 'dc'])


# Use my Michael Nunez's own recovery function

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


# Plot the results
plt.figure()
recovery(param_samples[:, :, 0].T.reshape(n_param_sets, n_samples, 1), true_params[:, 0].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Drift')

plt.figure()
recovery(param_samples[:, :, 1].T.reshape(n_param_sets, n_samples, 1), true_params[:, 1].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Boundary')

plt.figure()
recovery(param_samples[:, :, 2].T.reshape(n_param_sets, n_samples, 1), true_params[:, 2].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Non-decision time')

plt.figure()
recovery(param_samples[:, :, 3].T.reshape(n_param_sets, n_samples, 1), true_params[:, 3].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Relative Start Point')

plt.figure()
recovery(param_samples[:, :, 4].T.reshape(n_param_sets, n_samples, 1), true_params[:, 4].squeeze())
plt.xlabel('True')
plt.ylabel('Posterior')
plt.title('Diffusion Coefficient')
