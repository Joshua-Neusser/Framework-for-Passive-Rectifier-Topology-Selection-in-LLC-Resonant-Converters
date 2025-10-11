import numpy as np
import warnings 

def compute_dead_time(time, iDS, fsw, SimCycles):
    """Helper function to compute dead time (example implementation)."""
    try:
        start_idx = np.where(time > ((SimCycles - 1) + 0.5) / fsw)[0][0]
        next_on_idx = np.where(iDS[start_idx:] > 0.001)[0][0]
        dead_time = time[start_idx + next_on_idx] - time[start_idx]
        return dead_time
    except IndexError:
        warnings.warn("Could not compute dead time. Check simulation length and iDS waveform.", RuntimeWarning)
        return np.nan

def analyze(analysis, fsw, TimeStep, SimCycles, return_arrays=False): 
    """
    Analyze the results of a transient simulation for an LLC converter.

    Extracts key waveforms and calculates performance indicators.
    Optionally returns the full waveform arrays.

    Parameters
    ----------
    analysis : PySpice waveform object
        The result object returned by `simulator.transient(...)`.
    fsw : float
        Switching frequency in Hz.
    TimeStep : float
        Simulation time step in seconds.
    SimCycles : int
        Number of full switching cycles simulated.
    return_arrays : bool, optional
        If True, includes the full NumPy waveform arrays in the returned
        dictionary. Defaults to False (to save memory).

    Returns
    -------
    dict
        A dictionary containing calculated scalar metrics.
        If `return_arrays` is True, the dictionary also includes the full
        waveform arrays, keyed by their variable names (e.g., 'time', 'vab').

        Scalar Metrics (typically calculated over the last cycle(s)):
        - 'voutAVG': Average output voltage
        - 'iSecRMS': RMS secondary side current (iD1 + iD2)
        ... 
        - 'MaximumDeadTime': Estimated dead time before switch turn-on

        Waveform Arrays (only included if return_arrays=True):
        - 'time': Simulation time array
        - 'vab': Voltage across the switching bridge
        ...
        - 'iDS': Approximated switch current

    """

    # Extract and Derive Waveforms 
    time = np.array(analysis.time)
    vab = np.array(analysis['vab'])
    vpri= np.array(analysis['pri'])
    vout= np.array(analysis['vo'])
    iR= np.array(analysis['VLs_plus'])
    iLm= np.array(analysis['VLm_plus'])
    iD1= np.array(analysis['VD1_cathode'])
    iD2= np.array(analysis['VD2_cathode'])
    iSec= iD1+iD2
    iout= np.array(analysis['VRload_plus'])
    vcs= np.array(analysis['vab']-analysis['cs'])
    ico = np.array(analysis['Vco_plus'])
    iDS = np.where(vab < 0.1, -iR, 0)



    cycles_to_analyze = 20
    points_per_cycle = int((1/fsw) / TimeStep) if TimeStep > 0 else 0
    required_points = points_per_cycle * cycles_to_analyze

    if points_per_cycle > 0 and len(time) >= required_points:
        analysis_slice = slice(-required_points, None)
    else:
        analysis_slice = slice(None, None)
        if points_per_cycle > 0:
            warnings.warn(
                f"Simulation length ({len(time)} points) is < required for {cycles_to_analyze} cycles "
                f"({required_points} points). Metrics calculated over full simulation.",
                RuntimeWarning
            )
    # --- 3. Calculate Scalar Metrics ---
    metrics = {
        'voutAVG': np.mean(vout[analysis_slice]).astype(float) if vout[analysis_slice].size > 0 else np.nan,
        'iSecRMS': np.sqrt(np.mean(iSec[analysis_slice]**2)).astype(float) if iSec[analysis_slice].size > 0 else np.nan,
        'iD1PK' : np.max(iD1[analysis_slice]).astype(float) if iD1[analysis_slice].size > 0 else np.nan,
        'iD1RMS' : np.sqrt(np.mean(iD1[analysis_slice]**2)).astype(float) if iD1[analysis_slice].size > 0 else np.nan,
        'iSecPK': np.max(iSec[analysis_slice]).astype(float) if iSec[analysis_slice].size > 0 else np.nan,
        'iRRMS': np.sqrt(np.mean(iR[analysis_slice]**2)).astype(float) if iR[analysis_slice].size > 0 else np.nan,
        'iRPK': np.max(iR[analysis_slice]).astype(float) if iR[analysis_slice].size > 0 else np.nan,
        'iDSRMS': np.sqrt(np.mean(iDS[analysis_slice]**2)).astype(float) if iDS[analysis_slice].size > 0 else np.nan,
        'iDSPK': np.max(iDS[analysis_slice]).astype(float) if iDS[analysis_slice].size > 0 else np.nan,
        'iDSoff': iDS[-1].astype(float) if len(iDS) > 0 else np.nan,
        'vCsRMS': np.sqrt(np.mean(vcs[analysis_slice]**2)).astype(float) if vcs[analysis_slice].size > 0 else np.nan,
        'vCsPK': np.max(vcs[analysis_slice]).astype(float) if vcs[analysis_slice].size > 0 else np.nan,
        'ioutAVG' : np.mean(iout[analysis_slice]).astype(float) if iout[analysis_slice].size > 0 else np.nan,
        'iLmAVG' : np.mean(iLm[analysis_slice]).astype(float) if iLm[analysis_slice].size > 0 else np.nan,
        'vLmPK': np.max(vpri[analysis_slice]).astype(float) if vpri[analysis_slice].size > 0 else np.nan,
        'iCoRMS' : np.sqrt(np.mean(ico[analysis_slice]**2)).astype(float) if ico[analysis_slice].size > 0 else np.nan,
        'MaximumDeadTime': float(compute_dead_time(time, iDS, fsw, SimCycles)),
    }

    # Optionally Add Full Arrays to the Return Dictionary 
    if return_arrays:
        arrays = {
            'time': time,
            'vab': vab,
            'vpri': vpri,
            'vout': vout,
            'iR': iR,
            'iLm': iLm,
            'iD1': iD1,
            'iD2': iD2,
            'iSec': iSec,
            'iout': iout,
            'vcs': vcs,
            'ico': ico,
            'iDS': iDS
        }

        return {**metrics, **arrays}
    else:
        return metrics


# --- Example Usage ---
# Assume 'analysis_results' is the object from your PySpice simulation
# and you have fsw, TimeStep, SimCycles defined:

# Case 1: Get only metrics (default behavior, memory efficient)
# results_metrics_only = analyze(analysis_results, fsw=100e3, TimeStep=1e-8, SimCycles=50)
# print("--- Metrics Only ---")
# print(f"Average Vout: {results_metrics_only['voutAVG']:.4f} V")
# # print(results_metrics_only['time']) # This would cause a KeyError

# Case 2: Get metrics AND full arrays
# results_with_arrays = analyze(analysis_results, fsw=100e3, TimeStep=1e-8, SimCycles=50, return_arrays=True)
# print("\n--- Metrics and Arrays ---")
# print(f"Average Vout: {results_with_arrays['voutAVG']:.4f} V")
# time_array = results_with_arrays['time']
# print(f"Number of time points: {len(time_array)}")
