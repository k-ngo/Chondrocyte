import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate.odepack import odeint
from chondrocyte import Voltage_clamp
import functions
from params import params_dict

# Configs
t_final = 5000  # params_dict['t_final']  # 50000 ms
dt = 0.1  # params_dict['dt']
OA = True
t = np.linspace(0, t_final, int(t_final / dt))

original_scales = params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
                  params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
                  params_dict['g_Cl_b_bar'], params_dict['gBK']

V_0 = params_dict['V_0']
Na_i_0 = params_dict['Na_i_0']
K_i_0 = params_dict['K_i_0']
Ca_i_0 = params_dict['Ca_i_0']
H_i_0 = params_dict['H_i_0']
Cl_i_0 = params_dict['Cl_i_0']

a_ur_0 = params_dict['a_ur_0']
i_ur_0 = params_dict['i_ur_0']
vol_i_0 = params_dict['vol_i_0']
cal_0 = params_dict['cal_0']
Q_10 = params_dict['Q_10']
H_o = params_dict['H_o']

params_dict['clamp_Na_i'] = False
params_dict['clamp_K_i'] = False
params_dict['calmp_Ca_i'] = False
# params_dict['clamp_Cl_i'] = True

parameter_names = ['I_K_DR', 'I_NaK', 'I_NaCa', 'I_Ca_ATP', 'I_K_ATP', 'I_K_2pore', 'I_Na_b', 'I_K_b', 'I_Cl_b',
                   'I_BK']
solution_names = ['V', 'Na_i', 'K_i', 'Ca_i', 'H_i', 'Cl_i', 'a_ur', 'i_ur', 'vol_i', 'cal']

# Male control
y0 = (V_0, Na_i_0, K_i_0, Ca_i_0, H_i_0, Cl_i_0, a_ur_0, i_ur_0, vol_i_0, cal_0)
male_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
male_VV, current_dict = Voltage_clamp(male_solution)
male_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                          current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                          current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                          current_dict['I_BK']])

# Female control (epi)
params_dict['g_K_DR'] *= 0.96 / 1.20
params_dict['NCX_scale'] *= 1.07 / 1.10
params_dict['I_NaK_bar'] *= 0.87 / 0.92
params_dict['g_K_b_bar'] *= 0.42 / 0.82

female_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
female_VV, current_dict = Voltage_clamp(female_solution)
female_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

# Male OA
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = original_scales

params_dict['I_NaK_bar'] *= 2.2
params_dict['I_Ca_ATP_scale'] *= 2
params_dict['I_K_2pore_scale'] *= 0.2
params_dict['g_K_DR'] *= 8.3
params_dict['Q_10'] *= 3 / 1.3

male_OA_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
male_OA_VV, current_dict = Voltage_clamp(male_OA_solution)
male_OA_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

# Female OA
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = original_scales

params_dict['I_NaK_bar'] *= 2.2
params_dict['I_Ca_ATP_scale'] *= 2
params_dict['I_K_2pore_scale'] *= 0.2
params_dict['g_K_DR'] *= 8.3
params_dict['Q_10'] *= 3 / 1.3

params_dict['g_K_DR'] *= 0.96 / 1.20
params_dict['NCX_scale'] *= 1.07 / 1.10
params_dict['I_NaK_bar'] *= 0.87 / 0.92
params_dict['g_K_b_bar'] *= 0.42 / 0.82

female_OA_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
female_OA_VV, current_dict = Voltage_clamp(female_OA_solution)
female_OA_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

# Male OA (I_KDR block)
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = original_scales

params_dict['I_NaK_bar'] *= 2.2
params_dict['I_Ca_ATP_scale'] *= 2
params_dict['I_K_2pore_scale'] *= 0.2
params_dict['g_K_DR'] *= 8.3
params_dict['Q_10'] *= 3 / 1.3

# params_dict['g_K_DR'] *= 1 - 0.55  # treatment
params_dict['I_NaK_bar'] *= 1.4  # treatment

block_male_OA_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
block_male_OA_VV, current_dict = Voltage_clamp(male_OA_solution)
block_male_OA_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

# Female OA (I_KDR block)
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = original_scales

params_dict['I_NaK_bar'] *= 2.2
params_dict['I_Ca_ATP_scale'] *= 2
params_dict['I_K_2pore_scale'] *= 0.2
params_dict['g_K_DR'] *= 8.3
params_dict['Q_10'] *= 3 / 1.3

params_dict['g_K_DR'] *= 0.96 / 1.20
params_dict['NCX_scale'] *= 1.07 / 1.10
params_dict['I_NaK_bar'] *= 0.87 / 0.92
params_dict['g_K_b_bar'] *= 0.42 / 0.82

# params_dict['g_K_DR'] *= 1 - 0.65  # treatment
params_dict['I_NaK_bar'] *= 1.5  # treatment

block_female_OA_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
block_female_OA_VV, current_dict = Voltage_clamp(female_OA_solution)
block_female_OA_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 3))

axes[0].plot(t, male_solution[:, 0], 'slategray', label='Control')
axes[0].plot(t, male_OA_solution[:, 0], 'tomato', label='OA')
axes[0].plot(t, block_male_OA_solution[:, 0], 'royalblue', linestyle='dashed', label='OA (55% $I_{K\_DR}$ Block)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Voltage (mV)')
axes[0].legend()
axes[0].set_title('Healthy Male vs. OA')

axes[1].plot(t, female_solution[:, 0], 'slategray', label='Control')
axes[1].plot(t, female_OA_solution[:, 0], 'tomato', label='OA')
axes[1].plot(t, block_female_OA_solution[:, 0], 'royalblue', linestyle='dashed', label='OA (65% $I_{K\_DR}$ Block)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Voltage (mV)')
axes[1].legend()
axes[1].set_title('Healthy Female vs. OA')

for ax in axes.flat:
    ax.set_ylim([-99, -25])

# plt.suptitle('Membrane Potential')

plt.tight_layout()
figure_name = 'figures/NaK_male_vs_female_OA.png'
plt.savefig(figure_name, dpi=300)
print('Figure saved as:', figure_name)
plt.show()
