import numpy as np

# The 8 coefficients
k_c3 = -0.00000139
d_c3 = 0.00519030
k_c2 = 0.00002516
d_c2 = 0.00394802
k_c1 = 0.00008131
d_c1 = 0.05750570
k_c0 = -0.00167147
d_c0 = 0.72890888

def calcular_vf_deg3(corrente, temperatura):
    log_i = np.log(corrente)
    
    c3 = k_c3 * temperatura + d_c3
    c2 = k_c2 * temperatura + d_c2
    c1 = k_c1 * temperatura + d_c1
    c0 = k_c0 * temperatura + d_c0
    
    vf = c3 * (log_i**3) + c2 * (log_i**2) + c1 * log_i + c0
    return vf

# Example
corrente_op = 2     # Amperes
temp_op = 70      # Graus Celsius

tensao_direta = calcular_vf_deg3(corrente_op, temp_op)
perda = tensao_direta * corrente_op

print(f"Para I = {corrente_op}A e T = {temp_op}°C:")
print(f"  - Vf estimado: {tensao_direta:.4f} V")
print(f"  - Perda de condução estimada: {perda:.4f} W")