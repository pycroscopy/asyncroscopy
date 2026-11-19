import numpy as np
import json

first_order  = ['WD', 'C1', 'A1']
second_order = ['B2', 'A2']
third_order  = ['C3', 'S3', 'A3']
fourth_order = ['A4', 'D4', 'B4']
all_keys     = first_order + second_order + third_order + fourth_order

# C1, C3 are scalar — Tango server expects "KEY x" (one float).
# All others are 2-component — expects "KEY x y".
SCALAR_KEYS = {'C1', 'C3'}

thresholds = {
    'WD': 1e-3,   'C1': 5e-9,   'A1': 5e-9,
    'B2': 50e-9,  'A2': 100e-9,
    'C3': 0.7e-6, 'S3': 0.5e-6, 'A3': 0.5e-6,
    'A4': 5e-6,   'D4': 4e-6,   'B4': 5e-6,
}
factors = {
    'WD': 0.5,   'C1': 1.0,   'A1': 1.0,
    'B2': 0.75,  'A2': 0.75,
    'C3': 0.5, 'S3': 0.75, 'A3': 0.75,
    'A4': 0.5,   'D4': 0.5,   'B4': 0.3,
}
def acquire(tableau_type, corrector_proxy):
    """
    Run a tableau and return the aberrations dict.
    tableau_type: e.g. 'Fast 1', 'Enhanced 40'
    """
    raw = corrector_proxy.acquire_tableau(tableau_type,)
    return json.loads(raw)['result']['aberrations']


def correct(key, v, corrector_proxy):
    """
    Send a correction via the Tango proxy.
    Scalar keys (WD, C1, C3) → "KEY x"
    Vector keys              → "KEY x y"
    """
    v = list(v)
    if key in SCALAR_KEYS:
        corrector_proxy.correct_aberration(f"{key} {v[0]}")
    else:
        corrector_proxy.correct_aberration(f"{key} {v[0]} {v[1]}")


def correct_12(ab, corrector_proxy):
    """Correct 1st and 2nd order. Returns list of corrected keys."""
    corrected = []
    
    for key in first_order + second_order:
        if np.linalg.norm(ab.get(key, [0, 0])) > thresholds[key]:
            correct(key, [v * factors[key] for v in ab[key]], corrector_proxy)
            corrected.append(key)
    return corrected


def correct_3rd(ab, corrector_proxy):
    """
    Correct 3rd order. C3 gates S3/A3 — if C3 is bad, skip S3/A3.
    Returns list of corrected keys.
    """
    corrected = []
    for key in ['C3', 'S3', 'A3']:
        if np.linalgnorm(ab.get(key, [0, 0])) > thresholds[key]:
            correct(key, [v * factors[key] for v in ab[key]], corrector_proxy)
            corrected.append(key)
    return corrected


def correct_4th(ab, factor=1, corrector_proxy=None):
    """Correct 4th order with partial factor. Returns list of corrected keys."""
    corrected = []
    for key in fourth_order:
        if np.linalgnorm(ab.get(key, [0, 0])) > thresholds[key]:
            correct(key, [v * factors[key]*factor for v in ab[key]], corrector_proxy)
            corrected.append(key)
    return corrected

