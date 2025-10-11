from concurrent.futures import ProcessPoolExecutor, as_completed
from llc_sim import build_llc_circuit, simulate, analyze
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


def distributed_parallel_max(center_kHz, span_kHz, params, workers=8):

    half_span = span_kHz / 2
    f_min = center_kHz - half_span
    f_max = center_kHz + half_span

    fsw_points = np.linspace(f_min, f_max, workers)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(negative_vout, f_kHz, params): f_kHz for f_kHz in fsw_points}
        for future in as_completed(futures):
            f_kHz = futures[future]
            try:
                loss = future.result()
                results.append((f_kHz, -loss)) 
            except Exception as e:
                print(f"[Erro] fsw = {f_kHz:.3f} kHz → {e}")

    if not results:
        raise RuntimeError("All simulations failed.")

    best_fsw_kHz, best_vout = max(results, key=lambda x: x[1])
    return best_fsw_kHz * 1e3, best_vout 


def max_voltage(params: dict):

    """
    Finds the maximum output voltage of the LLC converter.
    1. Uses FHA to quickly estimate the frequency of peak gain.
    2. Uses a parallel simulation search (`distributed_parallel_max`) centered
       around the FHA estimate to find a more accurate maximum voltage and
       the corresponding frequency based on simulation.

    Args:
        params (dict): Dictionary containing the base circuit parameters.

    Returns:
        tuple: (fha_results, sim_results)
               fha_results (list): [estimated_peak_freq_Hz, estimated_peak_Vout] from FHA.
               sim_results (list): [simulated_peak_freq_Hz, simulated_peak_Vout] from parallel search.
    """

    f_vals, gain_vals = fha_estimate(params)
    idx_max = np.argmax(gain_vals)

    fsw_guess_Hz = f_vals[idx_max]
    Vmax_est = gain_vals[idx_max]

    fsw_opt_Hz, vout_opt = distributed_parallel_max(
        center_kHz=fsw_guess_Hz / 1e3,
        span_kHz=10,
        params=params,
        workers=8
)


    return [fsw_guess_Hz, Vmax_est], [fsw_opt_Hz, vout_opt]


__all__ = ["max_voltage"]