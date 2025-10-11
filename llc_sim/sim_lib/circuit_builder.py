from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from .transformer import TrxSubCir_CT


def build_llc_circuit(params: dict, Vtarget=None, Config  = 1) -> Circuit:

    """
    Build a full LLC resonant converter circuit using PySpice.

    Parameters
    ----------
    fsw : float
        Switching frequency in Hz.
    Vbus : float
        Input bus voltage [V].
    Cs : float
        Series resonant capacitor value [F].
    Ls : float
        Series resonant inductor value [H].
    Lm : float
        Magnetizing inductor value [H].
    Co : float
        Output capacitor value [F].
    Rload : float
        Load resistance [Ohms].
    n : float
        Transformer turns ratio (primary/secondary).
    Vtarget : float, optional
        Initial target voltage for the output capacitor [V].
    L1 : float, optional
        Transformer primary inductance
    Rc1 : float, optional
        Secondary primary inductance
    Rc2 : float, optional
        Copper resistance
        
    Returns
    -------
    Circuit
        A PySpice `Circuit` object representing the configured LLC converter,
        including input stage, resonant tank, transformer, rectifier, output filter,
        and current probes for key components.
    """

    fsw = params['fsw']
    Vbus = params['Vbus']
    CsValue = params['Cs']
    LsValue = params['Ls']
    LmValue = params['Lm']
    CoValue = params['Co']
    RL = params['Rload']
    n = params['n']
    L1= params.get('L1', 100)
    Rc1= params.get('Rc1', 0.01)
    Rc2= params.get('Rc2', 0.01)

    period = 1 / fsw 
    D = 0.5
    duty_cycle = D * period 
    circuit = Circuit('LLC')

    circuit.model('MyDiode', 'D', RON=.1, ROFF=1e6, VFWD=.5, VREV=2000)

    
    
    circuit.model('Switch', 'SW', ron=1e-6, roff=1e6 ) 

    if Config == 1:

        circuit.V(1, 'N001', circuit.gnd, Vbus)

        circuit.VCS('switch1', 'N001', 'vab', 'N003', circuit.gnd, model='Switch', initial_state='off')
        circuit.VCS('switch2', 'vab', circuit.gnd, 'N004', circuit.gnd, model='Switch', initial_state='on')

        
        circuit.PulseVoltageSource('pulse', 'N003', circuit.gnd,
                                    initial_value=0,   
                                    pulsed_value=5,    
                                    pulse_width = duty_cycle ,
                                    period = period,
                                    delay_time=0,    
                                    rise_time=20e-9,       
                                    fall_time=20e-9)    
        
        circuit.PulseVoltageSource('pulse_comp', 'N004', circuit.gnd,
                                    initial_value=5,   
                                    pulsed_value=0,    
                                    pulse_width = duty_cycle ,
                                    period = period,
                                    delay_time=0,    
                                    rise_time=20e-9,       
                                    fall_time=20e-9) 
        
        circuit.C('s', 'vab', 'cs', CsValue)
        circuit.L('s', 'cs', 'pri', LsValue)
        circuit.L('m', 'pri', circuit.gnd, LmValue)

        trx = TrxSubCir_CT('TRX_LLC', turn_ratio=n, primary_inductance=L1,
                       copper_resistance_primary=Rc1, copper_resistance_secondary=Rc2)
        circuit.subcircuit(trx)
        circuit.X(1, 'TRX_LLC', 'pri', circuit.gnd, 'secn1', circuit.gnd, 'secn2')

    
        circuit.Diode(1, 'secn1', 'vo', model='MyDiode')
        circuit.Diode(2, 'secn2', 'vo', model='MyDiode')
        circuit.C('o', 'vo', circuit.gnd, CoValue)
        circuit.R('load', 'vo', circuit.gnd, RL)
  
    elif Config == 2:



        circuit.V(1, 'N001','N002', Vbus)
        circuit.R('G', 'N002', circuit.gnd, 1e9)

                # Leg A: Q1 (HS), Q2 (LS)
        circuit.VCS('Q1', 'N001', 'va', 'ctrl_Q1', circuit.gnd, model='Switch')
        circuit.VCS('Q2', 'va', 'N002', 'ctrl_Q2', circuit.gnd, model='Switch')

        # Leg B: Q3 (HS), Q4 (LS)
        circuit.VCS('Q3', 'N001', 'vb', 'ctrl_Q3', circuit.gnd, model='Switch')
        circuit.VCS('Q4', 'vb', 'N002', 'ctrl_Q4', circuit.gnd, model='Switch')


        circuit.C('s', 'va', 'cs', CsValue)
        circuit.L('s', 'cs', 'pri', LsValue)
        circuit.L('m', 'pri', 'vb', LmValue)
        

        trx = TrxSubCir_CT('TRX_LLC', turn_ratio=n, primary_inductance=L1,
                        copper_resistance_primary=Rc1, copper_resistance_secondary=Rc2)
        circuit.subcircuit(trx)
        circuit.X(1, 'TRX_LLC', 'pri', 'vb', 'secn1', circuit.gnd, 'secn2')

    
        circuit.Diode(1, 'secn1', 'vo', model='MyDiode')
        circuit.Diode(2, 'secn2', 'vo', model='MyDiode')
        circuit.C('o', 'vo',circuit.gnd, CoValue)
        circuit.R('load', 'vo', circuit.gnd, RL)

        # Define vab = va - vb using VCVS
        circuit.VoltageControlledVoltageSource('vab_def', 'vab', circuit.gnd, 'va', 'vb', voltage_gain=1)

        
        dead_time = 100e-9 # Example dead time of 50 nanoseconds
        # on_time is slightly less than half the period
        on_time = (period / 2) - dead_time

        # --- Corrected Control Signal Definitions ---

        # Q1 and Q4 are ON for the first half of the period
        circuit.PulseVoltageSource('ctrl_Q1', 'ctrl_Q1', circuit.gnd,
                                initial_value=5, pulsed_value=0,
                                delay_time=0, period=period,
                                pulse_width=on_time, rise_time=20e-9, fall_time=20e-9)

        circuit.PulseVoltageSource('ctrl_Q4', 'ctrl_Q4', circuit.gnd,
                                initial_value=0, pulsed_value=5,
                                delay_time=0, period=period, # In phase with Q1
                                pulse_width=on_time, rise_time=20e-9, fall_time=20e-9)


        # Q2 and Q3 are ON for the second half of the period
        circuit.PulseVoltageSource('ctrl_Q2', 'ctrl_Q2', circuit.gnd,
                                initial_value=5, pulsed_value=0,
                                delay_time=period/2, period=period,
                                pulse_width=on_time, rise_time=20e-9, fall_time=20e-9)

        circuit.PulseVoltageSource('ctrl_Q3', 'ctrl_Q3', circuit.gnd,
                                initial_value=0, pulsed_value=5,
                                delay_time=period/2, period=period, # In phase with Q2
                                pulse_width=on_time, rise_time=20e-9, fall_time=20e-9)

        
    
    else:
        circuit.PulseVoltageSource("Ideal", 'vab', circuit.gnd,
                               initial_value=0,
                               pulsed_value=Vbus,
                               pulse_width=1/(2*fsw),
                               period=1/fsw,
                               rise_time=20e-9,
                               fall_time=20e-9)
    
        circuit.C('s', 'vab', 'cs', CsValue)
        circuit.L('s', 'cs', 'pri', LsValue)
        circuit.L('m', 'pri', circuit.gnd, LmValue)

        trx = TrxSubCir_CT('TRX_LLC', turn_ratio=n, primary_inductance=L1,
                       copper_resistance_primary=Rc1, copper_resistance_secondary=Rc2)
        circuit.subcircuit(trx)
        circuit.X(1, 'TRX_LLC', 'pri', circuit.gnd, 'secn1', circuit.gnd, 'secn2')

    
        circuit.Diode(1, 'secn1', 'vo', model='MyDiode')
        circuit.Diode(2, 'secn2', 'vo', model='MyDiode')
        circuit.C('o', 'vo', circuit.gnd, CoValue)
        circuit.R('load', 'vo', circuit.gnd, RL)
        


    circuit.Ls.plus.add_current_probe(circuit)
    circuit.Lm.plus.add_current_probe(circuit)
    circuit.Rload.plus.add_current_probe(circuit)
    circuit.D1.plus.add_current_probe(circuit)
    circuit.D2.plus.add_current_probe(circuit)
    circuit.Co.plus.add_current_probe(circuit)

    


    '''
    if Vtarget is not None:

        circuit.PulseVoltageSource('pulse_ctrl', 'v_ctrl', circuit.gnd,
                                   initial_value=5,   
                                   pulsed_value=0,    
                                   pulse_width = 100,
                                   period = 1000,
                                   delay_time=0.5/fsw,    
                                   rise_time=20e-9,       
                                   fall_time=20e-9)     


        circuit.V('target', 'v_target_node', circuit.gnd, Vtarget)

        circuit.S('precharge_switch', 'v_target_node', 'vo', 'v_ctrl', circuit.gnd, model='Switch')
    '''
    
    return circuit


