from multiprocessing import Pool, set_start_method
import numpy as np
import pandas as pd
import emcee
from scipy.stats import norm, uniform, beta
import pickle

import os
import corner
import warnings
import wfdb as wf
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm, gamma
from scipy.signal import butter, filtfilt, resample, find_peaks

warnings.simplefilter(action='ignore', category=FutureWarning)


EPS_HBO2_660 = 319.6
EPS_HB_660 = 3226.56

EPS_HBO2_940 = 1214
EPS_HB_940 = 693.44

def AC(signal ,peaks):
    ac = np.zeros(len(peaks)-1)
    for i in range (0, len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        ac[i] = np.max(signal[start:end]) - np.min(signal[start:end])
    return ac

def DC(signal ,peaks):
    dc = np.zeros(len(peaks)-1)
    for i in range (0, len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        dc[i] = np.mean(signal[start:end])
    return dc

def R(ac_ir, ac_red, dc_ir, dc_red):
    r = np.zeros(len(ac_ir))
    for i in range(len(ac_ir)):
        r[i] =  (ac_red[i] / dc_red[i])/ (ac_ir[i] / dc_ir[i]) 
    return r

def extract_beats(ppg, f_ppg, min_time_between=0.4):
    """
    Arguments:
    ----------
    ppg: np.ndarray
        A one dimensional time series of ppg data (red or ir).
    f_ppg: int
        The sampling frequency (Hz).
    min_time_between: float
        The minimal time between two heartbeats.

    Returns:
    --------
    - peaks: np.ndarray
        The indices of the heartbeats peaks in the ppg time series.
    """
    min_number_between = int(min_time_between * f_ppg)
    peaks, _ = find_peaks(ppg, distance=min_number_between)
    return peaks
def meanR(SpO2, parameters):
    ePS_HBO2_660, ePS_HB_660, ePS_HBO2_940, ePS_HB_940 = parameters
    num = SpO2 * ePS_HBO2_660 + (1 - SpO2) * ePS_HB_660
    den = SpO2 * ePS_HBO2_940 + (1 - SpO2) * ePS_HB_940
    return num / den
def log_likelihood(parameters, r):
    sigma_square,ePS_HBO2_660, ePS_HB_660, ePS_HBO2_940, ePS_HB_940= parameters [:5]
    SpO2  = parameters[5:]
    SpO2_norm = SpO2 /100
    r_mean = meanR(SpO2_norm, [ePS_HBO2_660, ePS_HB_660, ePS_HBO2_940, ePS_HB_940])
    return np.sum(norm.logpdf(r,loc = r_mean,scale=np.sqrt(sigma_square)))
def log_prior(parameters):
    sigma_square = parameters[0]
    ePS_HBO2_660 = parameters[1]
    ePS_HB_660 = parameters[2]
    ePS_HBO2_940 = parameters[3]
    ePS_HB_940 = parameters[4]
    SpO2_scale = parameters[5:]/100
    log_sigma_square = uniform.logpdf(sigma_square, 1e-8, 0.2)
    log_ePS_HBO2_660 = norm.logpdf(ePS_HBO2_660,loc =EPS_HBO2_660,scale=100)
    log_ePS_HB_660 = norm.logpdf(ePS_HB_660,loc = EPS_HB_660,scale=100)
    log_ePS_HBO2_940 = norm.logpdf(ePS_HBO2_940,loc= EPS_HBO2_940,scale =100)
    log_ePS_HB_940 = norm.logpdf(ePS_HB_940,loc = EPS_HB_940,scale =100)
    SpO2_clip = np.clip(SpO2_scale, 1e-8, 1 - 1e-8)
    log_SpO2 = np.sum(beta.logpdf(SpO2_clip, 8, 2))

    return log_sigma_square + log_ePS_HBO2_660 + log_ePS_HB_660 + log_ePS_HBO2_940 + log_ePS_HB_940 + log_SpO2
def log_posterior(parameters, r):
    if np.any(parameters <= 0):
        return -np.inf
    return log_likelihood(parameters, r) + log_prior(parameters)

encounter_id = "c5dd95c1ac9fc618cab2e940096089c6a91be58206fa6fc6a1375c69c4922779"
f_spo2 = 2

start = 5 * 60

saturation = pd.read_csv(f'data/waveforms/{encounter_id[0]}/{encounter_id}_2hz.csv')
spo2 = saturation['dev59_SpO2'].to_numpy()[start * f_spo2:]
t_spo2 = np.arange(spo2.shape[0]) / (60 * f_spo2)
f_ppg = 86

start = (5 - 2.8) * 60

ppg, ppg_info = wf.rdsamp(f'data/waveforms/{encounter_id[0]}/{encounter_id}_ppg')
ppg = ppg[int(start * f_ppg):]

ir = ppg[:, 0]
red = ppg[:, 1]
t_ppg = np.arange(len(red)) / (60 * f_ppg)

peaks = extract_beats(red, f_ppg)
ac_red = AC(red, peaks)
ac_ir = AC(ir, peaks)
dc_red = DC(red, peaks)
dc_ir = DC(ir, peaks)

r = R(ac_ir, ac_red, dc_ir, dc_red)

r_resampled = resample(r, len(spo2))

r_200 = r_resampled[::200]
spo2_200 = spo2[::200]
t_200 = t_spo2[::200]
    

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    set_start_method('spawn', force=True)
    with Pool() as pool:
        initial_parameters = np.concatenate([np.array([0.1, EPS_HBO2_660,EPS_HB_660,EPS_HBO2_940,EPS_HB_940]), spo2_200])
        pos = initial_parameters + 0.1*np.random.randn(100, 35)
        pos = np.abs(pos)

        nwalkers, ndim = pos.shape
        print(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(r_200,),pool=pool)
        sampler.run_mcmc(pos, 25000000//nwalkers, progress=True)
    filename = "sampler.pkl"
    with open(filename, "wb") as f:
        pickle.dump(sampler, f)

    print(f"Sampler saved to {filename}")
    