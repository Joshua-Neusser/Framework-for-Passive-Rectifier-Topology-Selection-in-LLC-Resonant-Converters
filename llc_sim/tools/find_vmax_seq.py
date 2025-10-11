from llc_sim import build_llc_circuit, simulate, analyze
from scipy.optimize import minimize_scalar
import numpy as np

SimCycles = 150
TimeStep = 10e-9


def fha_gain(omega, A, Qe, N, params):
    gain = 1 / (2 * N) * 1 / (1 + A - A / omega**2 + 1j * (omega - 1 / omega) * Qe)
    return np.abs(gain) * params['Vbus']


def fha_estimate(params, num_points=1000):
    Ls = params["Ls"]
    Lm = params["Lm"]
    Cs = params["Cs"]
    Rload = params["Rload"]
    N = params["n"]

    A = Ls / Lm
    Zs = np.sqrt(Ls / Cs)
    Re = (8 / np.pi**2) * (N**2) * Rload
    Qe = Zs / Re
    wr = 1 / np.sqrt(Ls * Cs)
    omega_vals = np.linspace(4, 1000e3, num_points) / wr

    gain_vals = fha_gain(omega_vals, A, Qe, N, params)
    f_vals = omega_vals * wr / (2 * np.pi)
    return f_vals, gain_vals


def negative_vout(fsw_kHz, params):
    fsw = fsw_kHz * 1e3
    sim_params = params.copy()
    sim_params["fsw"] = fsw

    try:
        circuit = build_llc_circuit(sim_params)
        analysis = simulate(circuit, fsw, TimeStep, SimCycles, 0)
        vout = analyze(analysis, fsw, TimeStep, SimCycles).get("voutAVG", 0)
    except Exception as e:
        print(f"[Erro] fsw = {fsw_kHz:.3f} kHz → {e}")
        return 1e6
    return -vout 


def max_voltage_seq(params: dict)-> tuple[tuple[float, float], tuple[float, float]]:

    """Calculate max vout for given LLC params."""

    f_vals, gain_vals = fha_estimate(params)

    idx_max = np.argmax(gain_vals)

    fsw_guess_Hz = f_vals[idx_max]
    Vmax_est = gain_vals[idx_max]

    fsw_guess_kHz = fsw_guess_Hz / 1e3
    result = minimize_scalar(
        lambda fsw_kHz: negative_vout(fsw_kHz, params),
        bounds=(fsw_guess_kHz - 10, fsw_guess_kHz + 10),
        method='bounded',
        options={'xatol': 0.1}
    )
    fsw_opt_Hz = result.x * 1e3
    vout_opt = -result.fun
    return [fsw_guess_Hz, Vmax_est], [fsw_opt_Hz, vout_opt]

__all__ = ["max_voltage_seq"]