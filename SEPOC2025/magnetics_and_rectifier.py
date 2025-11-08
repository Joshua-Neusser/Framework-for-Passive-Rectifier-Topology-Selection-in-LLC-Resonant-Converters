import json
import math
import numpy as np
import pandas as pd
import pathlib
from typing import Sequence, Dict, Any
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

class MagneticDesigner:
    # Constants and defaults
    CORE_DB_FILENAME        = "cores_shapes_params.ndjson"
    DIODE_DB_FILENAME       = "diode_database.json"
    DEFAULT_DIAMETERS_MM    = [0.1]                 # Possible strand diameters
    SIGMA_CU                = 5.96e7                # Copper conductivity @20°C
    MU_0                    = 4 * math.pi * 1e-7    # Air permeability


    # Finds the databases and loads the cores and diodes
    def __init__(self, path_to_db_folder: str = "."):
        
        base_path = pathlib.Path.cwd()
        
        db_folder_path = base_path / path_to_db_folder
        
        self.core_db_path  = db_folder_path / self.CORE_DB_FILENAME
        self.diode_db_path = db_folder_path / self.DIODE_DB_FILENAME

        self._all_cores_raw = self._load_cores()
        self._diodes_db     = self._DiodeDatabase(self.diode_db_path)

        print(f"Cores loaded: {len(self._all_cores_raw)}. Diodes loaded: {len(self._diodes_db._diodes)}.")



    ##### Runs the design routines and display the results
    def display_comparison_report(self, tr_params: Dict, ls_params: Dict, operating_points: Sequence[Dict], TimeStep: int):

        fw_results = self.transformer_design_and_loss_calculation(tr_params, 'FW', operating_points, TimeStep)
        fb_results = self.transformer_design_and_loss_calculation(tr_params, 'FB', operating_points, TimeStep)
        ls_results = self.run_inductor_design_and_loss(ls_params, operating_points)

        # Design results table
        design_data = {
            'Parameter': [
                'Core Model', 'Ae (mm²)', 'Ve (mm³)', 'Primary Turns', 'Secondary Turns',
                'Pri Litz Wire', 'Sec Litz Wire', 'Window Fill (%)', '',
                'Diode Model', 'Vmax (V)', 'Imax (A)', 'Cost (USD)'
            ],
            'Transformer (FW)': [
                fw_results['design']['core'], f"{fw_results['design']['Ae']*1e6:.1f}",
                f"{fw_results['design']['Ve']*1e9:.1f}",
                f"{fw_results['design']['Np_turns']:.0f}", f"{fw_results['design']['Ns_turns']:.0f}",
                f"{fw_results['design']['Np_strands']:.0f} # {fw_results['design']['d_strand']*1e3:.1f}mm",
                f"{fw_results['design']['Ns_strands']:.0f} # {fw_results['design']['d_strand']*1e3:.1f}mm",
                f"{(fw_results['design']['window_used'] / fw_results['design']['Aw'] * 100):.2f}", '',
                fw_results['rectifier']["Model/Name"], fw_results['rectifier']["Vmax (V)"],
                fw_results['rectifier']["ImaxAVG (A)"], fw_results['rectifier']["Cost (1u) (USD)"]
            ],
            'Transformer (FB)': [
                fb_results['design']['core'], f"{fb_results['design']['Ae']*1e6:.1f}",
                f"{fb_results['design']['Ve']*1e9:.1f}",
                f"{fb_results['design']['Np_turns']:.0f}", f"{fb_results['design']['Ns_turns']:.0f}",
                f"{fb_results['design']['Np_strands']:.0f} # {fb_results['design']['d_strand']*1e3:.1f}mm",
                f"{fb_results['design']['Ns_strands']:.0f} # {fb_results['design']['d_strand']*1e3:.1f}mm",
                f"{(fb_results['design']['window_used'] / fb_results['design']['Aw'] * 100):.2f}", '',
                fb_results['rectifier']["Model/Name"], fb_results['rectifier']["Vmax (V)"],
                fb_results['rectifier']["ImaxAVG (A)"], fb_results['rectifier']["Cost (1u) (USD)"]
            ],
            'Inductor': [
                ls_results['design']['core'], f"{ls_results['design']['Ae']*1e6:.1f}",
                f"{ls_results['design']['Ve']*1e9:.1f}",
                f"{ls_results['design']['N_turns']:.0f}", "",
                f"{ls_results['design']['N_strands']:.0f} # {ls_results['design']['d_strand']*1e3:.1f}mm", "",
                f"{(ls_results['design']['window_used'] / ls_results['design']['Aw'] * 100):.2f}", '',
                "", "", "", ""
            ]
        }
        design_df = pd.DataFrame(design_data)

        print("\n" + "="*60)
        print(" " * 24 + "Designs Comparison")
        print("="*60)
        print(design_df.to_string(index=False))
        print("\n" * 2)

        fw_rows = []
        fb_rows = []

        for i, op in enumerate(operating_points):
            fw_total_loss = (fw_results['losses'][i]['winding_loss_mw'] +
                             fw_results['losses'][i]['core_loss_mw'] +
                             fw_results['losses'][i]['rectifier_loss_mw'])
            
            fb_total_loss = (fb_results['losses'][i]['winding_loss_mw'] +
                             fb_results['losses'][i]['core_loss_mw'] +
                             fb_results['losses'][i]['rectifier_loss_mw'])
            
            fw_rows.append({
                'Operating Point': op['case'],
                'Winding Loss (mW)': f"{fw_results['losses'][i]['winding_loss_mw']:.1f}",
                'Core Loss (mW)': f"{fw_results['losses'][i]['core_loss_mw']:.1f}",
                'Rectifier Loss (mW)': f"{fw_results['losses'][i]['rectifier_loss_mw']:.1f}",
                'Total Loss (mW)': f"{fw_total_loss:.1f}"
            })

            fb_rows.append({
                'Operating Point': op['case'],
                'Winding Loss (mW)': f"{fb_results['losses'][i]['winding_loss_mw']:.1f}",
                'Core Loss (mW)': f"{fb_results['losses'][i]['core_loss_mw']:.1f}",
                'Rectifier Loss (mW)': f"{fb_results['losses'][i]['rectifier_loss_mw']:.1f}",
                'Total Loss (mW)': f"{fb_total_loss:.1f}"
            })

        fw_losses_df = pd.DataFrame(fw_rows)
        fb_losses_df = pd.DataFrame(fb_rows)

        print("="*83)
        print(" " * 22 + "Losses Table - Full-Wave (FW) Rectfier")
        print("="*83)
        print(fw_losses_df.to_string(index=False))
        print("\n")

        print("="*83)
        print(" " * 21 + "Losses Table - Full-Bridge (FB) Rectfier")
        print("="*83)
        print(fb_losses_df.to_string(index=False))
        print("\n")


        self._plot_loss_comparison(fw_results, fb_results, operating_points)

        return fw_results, fb_results, ls_results


    ##### Design routines and losses calculation
    def transformer_design_and_loss_calculation(self, tr_params: Dict, design_type: str, OperatingPoints: Sequence[Dict], TimeStep: int):
        

        # Input data for the transformer design and diode selection
        n           = tr_params['n']
        Kw          = tr_params['Kw']
        J           = tr_params['J']
        Bmax        = tr_params['Bmax']
        diode_temp  = tr_params.get('diode_temp', 80)
        vd_margin   = tr_params.get('vd_margin', 1.05)
        id_margin   = tr_params.get('id_margin', 1.05)
        sort_by     = tr_params.get('sort_by', 'vf')


        if design_type == 'FW':                                         
            Kp      = n / (2+n)                                             # Primary winding core window fill factor
            Is_max  = max(op["iSecRMS"] * 0.7071 for op in OperatingPoints) # RMS current in one secondary winding
            Vd_max  = max(op["voutAVG"] * 2 for op in OperatingPoints)      # Diodes voltage
            Id_max  = max(op["ioutAVG"]/2 for op in OperatingPoints)        # Avg diodes current
        
        elif design_type == 'FB':
            Kp      =  n / (1+n)    
            Is_max  = max(op["iSecRMS"] for op in OperatingPoints)
            Vd_max  = max(op["voutAVG"] for op in OperatingPoints) 
            Id_max  = max(op["ioutAVG"]/2 for op in OperatingPoints)
            
        else: raise ValueError("design_type must be 'FW' or 'FB'")
        
        Ip_max  = max(op["iRRMS"] for op in OperatingPoints)    # Maximum primary side current of all OPs
        cores   = self._filter_cores(family=["pq"])             # Selected core family
        
        BiggestCore, max_Ae = None, 0

        # Iterates the OPs and finds the biggest needed core
        for op in OperatingPoints:
            spec = {'i_rms_p':  Ip_max, 
                    'i_rms_s':  Is_max, 
                    'v_pri':    op["vLmPK"], 
                    'fsw_min':  op['fsw_khz'] * 1e3,
                    'n':        n, 
                    'Kp':       Kp, 
                    'Kw':       Kw, 
                    'J':        J, 
                    'Bmax':     Bmax}
            
            # For each OP, the "transformer designer" function is called
            DesignedTransformer = self.design_transformer(spec, topology=design_type, cores=cores)
            
            # Makes sure that the final Designed transformer is the biggest from all OPs
            if DesignedTransformer['Ae'] >= max_Ae:
                max_Ae, BiggestCore = DesignedTransformer['Ae'], DesignedTransformer
        

        DesignedTransformer = BiggestCore

        # This function picks the design transformer core from the database, so any parameter can be pulled to be used later
        SelectedCore        = self._get_core_by_name(DesignedTransformer['core']) 

        # Based on the worst case scenario for de rectifier, the diode selection function is called
        SelectedDiode       = self._diodes_db.design_diode_selection(Vd_max, Id_max, vd_margin, id_margin, sort_by)

        # Constants for loss calculation
        AmbientTemperature, PorosityFactor, CoreMaterial = 40, 2.5, 'N87'   

        # Considered conductor diameter of each side
        ConductorDiameter_Primary     = (2*math.sqrt((DesignedTransformer['St_p']*PorosityFactor)/math.pi))   
        ConductorDiameter_Secondary  = (2*math.sqrt((DesignedTransformer['St_s']*PorosityFactor)/math.pi))

        # Number of layers of each side
        NumLayers_Primary = DesignedTransformer['Np_turns']/((2*SelectedCore['dims']['D']-2e-3)/ConductorDiameter_Primary) if (2*SelectedCore['dims']['D']-2e-3) != 0 else float('inf')
        NumLayers_Secondary = (DesignedTransformer['Ns_turns']*(2 if design_type=='FW' else 1))/((2*SelectedCore['dims']['D']-2e-3)/ConductorDiameter_Secondary) if (2*SelectedCore['dims']['D']-2e-3) != 0 else float('inf')
       
        # Mean Length of Turn of each side
        MLT_Primary = (NumLayers_Primary*ConductorDiameter_Primary*0.5+1e-3+(SelectedCore['dims']['F'])/2)*2*math.pi
        MLT_Secondary = (NumLayers_Secondary*ConductorDiameter_Secondary*0.5+1e-3+NumLayers_Primary*ConductorDiameter_Primary+(SelectedCore['dims']['F'])/2)*2*math.pi


        # Losses calculation
        LossResults = []
        for op in OperatingPoints:

            WindingLoss_Primary = self.WindingLoss_SimplifiedSullivan(op['iRRMS'], DesignedTransformer['Np_turns'], DesignedTransformer['Np_strands'], DesignedTransformer['d_strand'], op['fsw_khz']*1e3, (2*SelectedCore['dims']['D']-2e-3), MLT_Primary, 40)
            if design_type == 'FW':
                WindingLoss_Secondary = 2*self.WindingLoss_SimplifiedSullivan(op["iSecRMS"]/2, DesignedTransformer['Ns_turns'], DesignedTransformer['Ns_strands'], DesignedTransformer['d_strand'], op['fsw_khz']*1e3, (2*SelectedCore['dims']['D']-2e-3), MLT_Secondary, 40)
            else: # FB
                WindingLoss_Secondary = self.WindingLoss_SimplifiedSullivan(op["iSecRMS"], DesignedTransformer['Ns_turns'], DesignedTransformer['Ns_strands'], DesignedTransformer['d_strand'], op['fsw_khz']*1e3, (2*SelectedCore['dims']['D']-2e-3), MLT_Secondary, 40)
            WindingLoss_Total = WindingLoss_Primary + WindingLoss_Secondary
        
            # Time and voltage arrays from the last simulated cycle, to be used in the core loss calculation.
            start_idx = np.argmin(np.abs(op['time'] - (op['time'][-1] - 1/(op['fsw_khz']*1e3))))
            t = op['time'][start_idx:] # Time array
            V = op['vpri'][start_idx:] # Transformer primary side voltage array

            CoreLoss = self.TransformerCoreLoss_WcSE(V,t,DesignedTransformer['Np_turns'],op['fsw_khz']*1e3,SelectedCore['Ae'],SelectedCore['Ve'],CoreMaterial)
            
            CoreTemp = self.CoreTemperature_Simple(CoreLoss, SelectedCore['Ve'], AmbientTemperature)

            # Rectifier Loss (see expression (2))
            log_i = np.log(op['ioutAVG']/2)

            c3 = SelectedDiode["k_c3"] * diode_temp + SelectedDiode["d_c3"]
            c2 = SelectedDiode["k_c2"] * diode_temp + SelectedDiode["d_c2"]
            c1 = SelectedDiode["k_c1"] * diode_temp + SelectedDiode["d_c1"]
            c0 = SelectedDiode["k_c0"] * diode_temp + SelectedDiode["d_c0"]
        
            vf = c3 * (log_i**3) + c2 * (log_i**2) + c1 * log_i + c0

            if design_type == 'FW':
                RectifierLoss = vf * op['ioutAVG']/2 * 2
            else:
                RectifierLoss = vf * op['ioutAVG']/2 * 4


            LossResults.append({'case':op['case'], 'winding_loss_mw':WindingLoss_Total*1e3, 'core_loss_mw':CoreLoss*1e3,
                                 'rectifier_loss_mw':RectifierLoss*1e3, 'total_loss_mw':(WindingLoss_Total+CoreLoss+RectifierLoss)*1e3,
                                 'core_temp_c':CoreTemp})
            
        return {'design':DesignedTransformer, 'losses':LossResults, 'rectifier':SelectedDiode}

    def run_inductor_design_and_loss(self, ls_params: Dict, OperatingPoints: Sequence[Dict]):
        
        # Input data
        Ls      = ls_params['Ls']
        Kw      = ls_params['Kw']
        J       = ls_params['J']
        Bmax    = ls_params['Bmax']

        cores = self._filter_cores(family=["e"]) # Selected family of cores
        
        BiggestCore, max_Ae = None, 0

        # For each OP, a inductor is designed, the biggest one is selected
        for op in OperatingPoints:
            spec = {'i_rms':op["iRRMS"], 
                    'i_pk': op["iRPK"], 
                    'fsw':  op['fsw_khz']*1e3, 
                    'Ls':   Ls, 'Kw':Kw, 
                    'J':    J, 
                    'Bmax': Bmax}
            
            DesignedInductor = self.design_inductor(spec, cores=cores)

            if DesignedInductor['Ae'] >= max_Ae:
                max_Ae, BiggestCore = DesignedInductor['Ae'], DesignedInductor
        
        DesignedInductor = BiggestCore
        SelectedCore = self._get_core_by_name(DesignedInductor['core'])

        # Constants for loss calculation
        AmbientTemperature, PorosityFactor, CoreMaterial = 40, 2.5, 'N87'

        # Considered conductor diameter for number of layers calculation
        ConductorDiameter = (2*math.sqrt((DesignedInductor['St']*PorosityFactor)/math.pi))

        # Number of Conductor Layers
        NumLayers = DesignedInductor['N_turns']/((2*SelectedCore['dims']['D']-2e-3)/ConductorDiameter)
        
        # MLT calculation for different core families
        if SelectedCore['family'].lower() == 'e':
            MLT = (NumLayers*ConductorDiameter*0.5+1e-3)*2*math.pi+4*SelectedCore['dims']['F']
        if SelectedCore['family'].lower() == 'pq':
            MLT = (NumLayers*ConductorDiameter*0.5+1e-3+(SelectedCore['dims']['F']/2))*2*math.pi

        # Loss calculation
        LossResults = []
        for op in OperatingPoints:
            winding_loss = self.WindingLoss_SimplifiedSullivan(op['iRRMS'], DesignedInductor['N_turns'], DesignedInductor['N_strands'], DesignedInductor['d_strand'], op['fsw_khz']*1e3, (2*SelectedCore['dims']['D']-2e-3), MLT, 40)
            CoreLoss = self.InductorCoreLoss_SE(Ls, DesignedInductor['N_turns'], op["iRPK"], op['fsw_khz']*1e3, SelectedCore['Ae'], SelectedCore['Ve'], CoreMaterial)
            CoreTemp = self.CoreTemperature_Simple(CoreLoss, SelectedCore['Ve'], AmbientTemperature)
            
            LossResults.append({'case':op['case'], 'winding_loss_mw':winding_loss*1e3, 
                                 'core_loss_mw':CoreLoss*1e3, 'core_temp_c':CoreTemp})
        return {'design': DesignedInductor, 'losses': LossResults}


    def design_inductor(self, spec: Dict[str, float], cores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        
        # Design input from the simulation
        i_rms, i_pk, fsw = spec["i_rms"], spec["i_pk"], spec["fsw"]
        
        # Design input from the user
        Ls, Kw, J, Bmax = spec["Ls"], spec["Kw"], spec["J"], spec["Bmax"]
        
        # Minimum AeAw required
        AeAw_req = (Ls * i_rms * i_pk) / (Bmax * J * Kw)

        # Iterates the cores in the database until a suitable core is found
        for c in cores:
            Ae, Aw, Ab = c["Ae"], c["Aw"], c["Ab"] # Core effective parameters

            # Checks if the current core has the minimum AeAw required
            if Ae * Aw < AeAw_req: continue 

            # Minimum Number of turns required by the current core
            N_min = (Ls * i_pk) / (Ae * Bmax)

            # Copper cross-sectional area required
            St = i_rms / J

            # Fit check, considering a porosity factor of 40%
            if St * 2.5 * N_min + Ab > Kw * Aw: continue

            # Skin depth
            delta = (1/(math.pi*fsw*self.MU_0*self.SIGMA_CU))**0.5

            # Finds the biggest suitable strand diameter
            diam_m = max((d for d in map(lambda mm:mm*1e-3, self.DEFAULT_DIAMETERS_MM) if d<=delta), default=min(self.DEFAULT_DIAMETERS_MM)*1e-3)

            # Cross-sectional are of one strand
            A_strand = math.pi * (diam_m / 2)**2

            # Number of strands
            N_strands = math.ceil(St / A_strand)

            # Final fit check
            window_used = N_strands * A_strand*2.5 * N_min + Ab
            if window_used > Kw * Aw:    
                continue


            return {"core":c["name"], "Ve":c["Ve"], "Ae":Ae, "Aw":Aw, "Ab":Ab, "AeAw_req":AeAw_req, "N_turns":N_min,
                    "d_strand":diam_m, "N_strands":math.ceil(N_strands), "delta_skin":delta, 
                    "window_used":window_used, "window_max":Kw*Aw, "St":St}
        raise ValueError("None of the cores match the inductor design requirements.")

    def design_transformer(self, spec: Dict[str, float], topology: str, cores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        
        # Input data from the simulation
        i_rms_p, i_rms_s, v_pri, fsw_min = spec["i_rms_p"], spec["i_rms_s"], spec["v_pri"], spec["fsw_min"]
        
        # Input data from the user
        n, Kp, Kw, J, Bmax = spec["n"], spec["Kp"], spec["Kw"], spec["J"], spec["Bmax"]
        
        # Minimum AeAw required
        AeAw_req = (v_pri * i_rms_p) / (4 * J * Kw * Kp * Bmax * fsw_min)
        # Iterates the design until a suitable core is found, as described in Section V
        for c in cores:
            Ae, Aw, Ab = c["Ae"], c["Aw"], c["Ab"] # Core effective parameters

            # Checks if the current core has the minimum AeAw required
            if Ae * Aw < AeAw_req: continue

            # Calculates the minimum number of turns required by the current core
            Np_min = math.ceil(v_pri / (4 * fsw_min * Bmax * Ae))

            # Copper cross-sectional area required by the primary and secondary windings
            St_p = i_rms_p / J
            St_s = i_rms_s / J

            # Total number of turns of the secondary winding(s), depends on the rectifier type
            Ns_turns_check = (2 if topology=='FW' else 1)*(Np_min/n)

            # Window fit check with porosity factor = 40%
            if St_p*2.5*Np_min + St_s*2.5*Ns_turns_check + Ab > Kw*Aw: continue
            
            # Skin depth
            delta = (1/(math.pi*fsw_min*self.MU_0*self.SIGMA_CU))**0.5

            # Finds the biggest suitable strand diameter
            diam_m = max((d for d in map(lambda mm:mm*1e-3, self.DEFAULT_DIAMETERS_MM) if d<=delta), default=min(self.DEFAULT_DIAMETERS_MM)*1e-3)
            
            # Cross-sectional area of one strand
            A_strand = math.pi*(diam_m/2)**2

            # Number of strands
            Np_strands = math.ceil(St_p/A_strand)
            Ns_strands = math.ceil(St_s/A_strand)


            
            # Final fit check
            window_used = Np_strands*A_strand*2.5*Np_min + Ns_strands*A_strand*2.5*Ns_turns_check + Ab
            if window_used > Kw*Aw: continue


            return {"core":c["name"], "Ve":c["Ve"], "Ae":Ae, "Aw":Aw, "Ab":Ab, "AeAw_req":AeAw_req, "Np_turns":math.ceil(Np_min),
                    "Ns_turns":round(Np_min/n), "d_strand":diam_m, "Np_strands":Np_strands, "Ns_strands":Ns_strands,
                    "delta_skin":delta, "window_used":window_used, "window_max":Kw*Aw, "St_p":St_p, "St_s":St_s}
        
        raise ValueError("None of the cores match the transformer design requirements.")




    ##### Simple models
    def WindingLoss_SimplifiedSullivan(self, iRMS, N, Nstrands, StrandDiameter, fsw, WindingHeight, MLT, Temperature):
        '''
        Descrição: Modelo de perdas no enrolamento simplificado do Sullivan

        Referência: C. R. Sullivan and R. Y. Zhang, "Simplified design method for litz wire," 2014 IEEE Applied Power Electronics Conference and Exposition - APEC 2014, Fort Worth, TX, USA, 2014

        Características:

        - Equação simples
        - Baseia-se na aproximação 1D do campo
        - Não considera o franjamento do fluxo (ou seja, não considera um gap)

        Inputs:

        - iRMS              = Corrente RMS no enrolamento
        - N                 = Número de voltas do enrolamento
        - Nstrands          = Número de strands (no caso de fio litz)
        - StrandDiameter    = Diâmetro de cada strand
        - WindingHeight     = Altura do enrolamento
        - MLT               = Comprimento médio de uma volta
        - Temperature       = Temperatura de operação

        Outputs:

        - Rdc               = Resistência DC
        - Rac               = Resistência AC
        - WindingLoss       = Perdas no enrolamento

        '''        

        # 0. Seção total de cobre para o Litz selecionado 
        WireTotalSection = Nstrands * (np.pi*(StrandDiameter/2)**2)

        # 0. Resistividade do cobre de acordo com a temperatura de operação (Tref = 20°C)
        CopperResistivity = 1.724e-8 * (1 + 0.00393*(Temperature - 20))

        # Skin Depth
        SkinDepth = np.sqrt(CopperResistivity/(np.pi * fsw * 4e-7 * np.pi))

        # AC resistance factor
        Fr = 1 + (((np.pi * Nstrands * N)**2 * StrandDiameter**6 ) / (192 * SkinDepth**4 * WindingHeight**2))

        Rdc = MLT * N * (CopperResistivity / WireTotalSection)
        Rac = Fr * Rdc
        WindingLoss = iRMS**2 * Rac 

        return WindingLoss

    def _GetSteinmetzCoeffs(self, fsw, Material):
        if Material == 'N87':
            if 50000 <= fsw <= 63000: return 9.04214, 1.32438, 2.61322
            if 63000 < fsw <= 79000: return 47.6963, 1.17714, 2.62825
            if 79000 < fsw <= 99000: return 11.9105, 1.3075, 2.65732
            if 99000 < fsw <= 126000: return 9.25949, 1.33577, 2.68223
            if 126000 < fsw <= 158000: return 0.202833, 1.6587, 2.67304
            if 158000 < fsw <= 600000: return 0.00821156, 1.91396, 2.62946 # Adicionar mais range preciso no futuro
        raise ValueError(f"Steinmetz coefficients not defined for fsw={fsw/1e3:.1f}kHz and material={Material}")

    def InductorCoreLoss_SE(self, L, N, iPk, fsw, Ae, Ve, Material):
        
        k, alpha, beta = self._GetSteinmetzCoeffs(fsw, Material)
        Bmax = (L * iPk) / (N * Ae)
        
        return k * (fsw**alpha) * (Bmax**beta) * Ve

    def TransformerCoreLoss_WcSE(self, V, t, N, fsw, Ae, Ve, Material):

        '''

        ----------------------------------- TRANSFORMER CORE LOSS -----------------------------------

        Descrição: Modelo de perdas no núcleo para ondas não senoidais WcSE

        Referência: W. Shen, F. Wang, D. Boroyevich and C. W. Tipton, "Loss Characterization and Calculation of Nanocrystalline Cores for High-frequency Magnetics Applications," 
        APEC 07 - Twenty-Second Annual IEEE Applied Power Electronics Conference and Exposition, Anaheim, CA, USA, 2007.

        Inputs:

        - V         = Array de valores de tensão de um período de chaveamento
        - t         = Array de valores de tempo de um período de chaveamento
        - N         = Número de voltas do primário
        - fsw       = Frequência de chaveamento
        - Ae        = Área efetiva do núcleo
        - Ve        = Volume efetivo do núcleo
        - Material  = Material do núcleo

        Output:

        - CoreLossTotal = Perdas no núcleo

        '''

        k, alpha, beta = self._GetSteinmetzCoeffs(fsw, Material)

        # 1. Cálculo do Bmax e Bavg
        # 1.1 Integrar numericamente v(t) para obter B(t)
        B_raw = (1/(N * Ae)) * cumulative_trapezoid(V, t, initial=0)  # initial=0 garante que o array de B tenha o mesmo tamanho do de V.

        # 1.2 Ajustar B(t) para que sua média seja zero
        B_offset = np.mean(B_raw)
        B_t = B_raw - B_offset

        # 1.3 Encontrar Bmax e Bavg
        Bmax = np.max(np.abs(B_t))
        
        # 1.4 Bavg é a média do valor absoluto da forma de onda de B
        Bavg = np.mean(np.abs(B_t))

        # 2. Cálculo do FWC (Warbitrary / Wsin)
        Warbitrary = Bavg / Bmax
        Wsin = 2 / np.pi
        FWC = Warbitrary / Wsin

        # 3. Calcular a perda de potência total no núcleo
        CoreLossTotal = FWC * k * (fsw ** alpha) * (Bmax ** beta) * Ve

        #print("Core Loss =", f"{(CoreLossTotal*1e3):.3f}", "mW")

        return CoreLossTotal

    def CoreTemperature_Simple(self, CoreLoss, Ve, Tambient):
        '''

        ----------------------------------- TRANSFORMER TEMPERATURE -----------------------------------

        Descrição: Modelo simples da temperatura do núcleo

        Referência: Sanjaya Maniktala, Switching power supplies A-Z, 2nd edition, Newnes, 2012, ISBN 978-0-12-386533-5, p. 155

        Inputs:

        - CoreLoss  = Perdas no núcleo
        - Ve        = Volume efetivo do núcleo (em cm³)
        - Tambient  = Temperatura ambiente

        Output:

        - CoreTemperature = Temperatura do núcleo

        '''


        CoreThermalResistance = 53 * (Ve * 1e6)**(-0.54)
        return Tambient + CoreThermalResistance * CoreLoss




    ##### Auxiliary functions
    def _nom(self, v):
        if isinstance(v, (int, float)): return v
        if isinstance(v, dict):
            if (n := v.get("nominal")) is not None: return n
            mn, mx = v.get("minimum"), v.get("maximum")
            if mn is not None and mx is not None: return 0.5 * (mn + mx)
            return mn if mn is not None else mx
        return None

    def _load_cores(self):
        cores = []
        if not self.core_db_path.exists():
            print(f"Warning: Database file not found in '{self.core_db_path}'")
            return []
        for line in self.core_db_path.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            d = json.loads(line)
            p = d.get("parameters", {})
            if not {"Ae", "Aw", "Ab", "Ve"} <= p.keys(): continue
            dims_nom = {k: self._nom(v) for k, v in d.get("dimensions", {}).items()}
            cores.append({"name": d["name"], "family": d["family"], **p, "dims": dims_nom})
        return sorted(cores, key=lambda c: c['Ae'] * c['Aw'])

    def _filter_cores(self, family: list = None):
        if family is None: return self._all_cores_raw
        family_lower = {f.lower() for f in family}
        return [c for c in self._all_cores_raw if c['family'].lower() in family_lower]
        
    def _get_core_by_name(self, name):
        for core in self._all_cores_raw:
            if core['name'].lower() == name.lower():
                return core
        raise ValueError(f"Core model '{name}' not found in the database.")

    def _plot_loss_comparison(self, fw_results: Dict, fb_results: Dict, operating_points: Sequence[Dict]):
            """Cria e exibe um gráfico de barras comparando as perdas."""

            op_labels = [op['case'] for op in operating_points]

            fw_losses = fw_results['losses']
            fb_losses = fb_results['losses']

            fw_winding = np.array([res['winding_loss_mw'] for res in fw_losses])
            fw_core    = np.array([res['core_loss_mw'] for res in fw_losses])
            fw_rect    = np.array([res['rectifier_loss_mw'] for res in fw_losses])

            fb_winding = np.array([res['winding_loss_mw'] for res in fb_losses])
            fb_core    = np.array([res['core_loss_mw'] for res in fb_losses])
            fb_rect    = np.array([res['rectifier_loss_mw'] for res in fb_losses])

            fw_magnetic = fw_winding + fw_core
            fb_magnetic = fb_winding + fb_core
            fw_total = fw_magnetic + fw_rect
            fb_total = fb_magnetic + fb_rect
            
            x = np.arange(len(op_labels)) 
            width = 0.12  
            
            fig, ax = plt.subplots(figsize=(8, 6))
            

            edge_color = 'black'
            line_width = 1 

            ax.bar(x - 2.5*width, fw_magnetic, width, label='Transformer (FW)', color='#66b3ff', edgecolor=edge_color, linewidth=line_width)
            ax.bar(x - 1.5*width, fb_magnetic, width, label='Transformer (FB)', color='#004c99', edgecolor=edge_color, linewidth=line_width)

            ax.bar(x - 0.5*width, fw_rect, width, label='Rectifier (FW)', color='#99ff99', edgecolor=edge_color, linewidth=line_width)
            ax.bar(x + 0.5*width, fb_rect, width, label='Rectifier (FB)', color='#008000', edgecolor=edge_color, linewidth=line_width)
            
            ax.bar(x + 1.5*width, fw_total, width, label='Total (FW)', color='#ff9999', edgecolor=edge_color, linewidth=line_width)
            ax.bar(x + 2.5*width, fb_total, width, label='Total (FB)', color='#cc0000', edgecolor=edge_color, linewidth=line_width)


            ax.tick_params(axis='both', labelsize=12) 
            ax.set_ylabel('Loss (mW)', fontsize=14)
            ax.set_xlabel('Operating Point', fontsize=14)
            ax.set_title('Losses by Rectifier Topology and Operating Point', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(op_labels, rotation=0)
            ax.legend(loc='upper left')
            ax.grid(axis='y', linestyle=':', linewidth=1)
            ax.set_axisbelow(True) 

            fig.tight_layout() 

            plt.savefig('loss_comparison_chart.pdf', bbox_inches='tight')
            plt.show()


    class _DiodeDatabase:
        def __init__(self, file_path):
            self._diodes = self._load_diodes_from_json(file_path)
            self._diodes_by_model = {d['Model/Name']: d for d in self._diodes}

        def _load_diodes_from_json(self, file_path):
            if not file_path.exists():
                print(f"Warning: Diodes database file not found in '{file_path}'.")
                return []
            diodes_list = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): diodes_list.append(json.loads(line))
            return diodes_list

        def design_diode_selection(self, required_voltage, required_current, v_margin=1.05, i_margin=1.05, sort_by='vf'):
            target_voltage, target_current = required_voltage*v_margin, required_current*i_margin
            sort_keys = {'cost':'Cost (1u) (USD)','vf':'Vf (Inom, 25°C) (V)','trr':'Reverse Recovery Time (ns)'}
            if sort_by.lower() not in sort_keys: raise ValueError(f"Sort criteria '{sort_by}' invalid.")
            sort_key = sort_keys[sort_by.lower()]
            suitable_diodes = [d for d in self._diodes if d.get('Vmax (V)',0)>=target_voltage and d.get('ImaxAVG (A)',0)>=target_current]
            if not suitable_diodes: raise ValueError("None of the diodes match the design requirements.")
            return sorted(suitable_diodes, key=lambda d: d.get(sort_key) if d.get(sort_key) is not None else float('inf'))[0]

        def __getitem__(self, model_name):
            try: return self._diodes_by_model[model_name]
            except KeyError: raise KeyError(f"Diode '{model_name}' not found.")