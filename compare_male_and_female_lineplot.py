import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate.odepack import odeint
from chondrocyte import Voltage_clamp
import functions
from params import params_dict
import os

# Configs
t_final = 100000  # params_dict['t_final']  # 50000 ms
dt = 0.1  # params_dict['dt']
figure_name = os.path.join('figures', 'male_vs_female_OA.png')
t = np.linspace(0, t_final, int(t_final / dt))

# Set custom scales for different conditions
params_dict['I_NaK_scale'] = 0.35
params_dict['I_NaK_bar'] = params_dict['I_NaK_scale']*70.8253*params_dict['C_m']/params_dict['C_myo']

baseline_scales = [params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
                   params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
                   params_dict['g_Cl_b_bar'], params_dict['gBK']]

OA_scales = baseline_scales.copy()
OA_scales[1] *= 2.2  # I_NaK_bar
OA_scales[3] *= 2  # I_Ca_ATP_scale
OA_scales[5] *= 0.2  # I_K_2pore_scale
OA_scales[0] *= 8.3  # g_K_DR
OA_scales[4] *= 3 ** ((23 - 36) / 10) / 1.3 ** ((23 - 36) / 10)  # I_K_ATP, this is Q_10 changes integrated into sigma

female_scales = baseline_scales.copy()
female_scales[1] *= 0.87 / 0.92  # I_NaK_bar
female_scales[0] *= 0.96 / 1.20  # g_K_DR
female_scales[2] *= 1.07 / 1.10  # NCX_scale
female_scales[7] *= 0.42 / 0.82  # g_K_b_bar

female_OA_scales = female_scales.copy()
female_OA_scales[1] *= 2.2  # I_NaK_bar
female_OA_scales[3] *= 2  # I_Ca_ATP_scale
female_OA_scales[5] *= 0.2  # I_K_2pore_scale
female_OA_scales[0] *= 8.3  # g_K_DR
female_OA_scales[4] *= 3 ** ((23 - 36) / 10) / 1.3 ** ((23 - 36) / 10)  # I_K_ATP, this is Q_10 changes integrated into sigma

# Set universal parameters
params_dict['clamp_Na_i'] = False
params_dict['clamp_K_i'] = False
params_dict['calmp_Ca_i'] = False
# params_dict['clamp_Cl_i'] = True
parameter_names = ['I_KDR', 'I_NaK', 'I_NaCa', 'I_Ca-ATP', 'I_K-ATP', 'I_K-2pore', 'I_Na-b', 'I_K-b', 'I_Cl-b',
                   'I_KCa']
solution_names = ['V', 'Na_i', 'K_i', 'Ca_i', 'H_i', 'Cl_i', 'a_ur', 'i_ur', 'vol_i', 'cal']

########################################################################################################################
# Male control
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = baseline_scales

y0 = (params_dict['V_0'], params_dict['Na_i_0'], params_dict['K_i_0'], params_dict['Ca_i_0'], params_dict['H_i_0'],
      params_dict['Cl_i_0'], params_dict['a_ur_0'], params_dict['i_ur_0'], params_dict['vol_i_0'],
      params_dict['cal_0'])

male_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=1000)
male_VV, current_dict = Voltage_clamp(male_solution)
male_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                          current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                          current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                          current_dict['I_BK']])

########################################################################################################################
# Female control (epi)
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = female_scales

y0 = (params_dict['V_0'], params_dict['Na_i_0'], params_dict['K_i_0'], params_dict['Ca_i_0'], params_dict['H_i_0'],
      params_dict['Cl_i_0'], params_dict['a_ur_0'], params_dict['i_ur_0'], params_dict['vol_i_0'],
      params_dict['cal_0'])

female_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=1000)
female_VV, current_dict = Voltage_clamp(female_solution)
female_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

########################################################################################################################
# Male OA
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = OA_scales

# params_dict['C_m'] = 6.3 * 37.93 / 21.56
# params_dict['NCX_scale'] = params_dict['C_m'] / params_dict['C_myo']
# params_dict['I_NaK_bar'] = params_dict['I_NaK_scale'] * 70.8253 * params_dict['C_m'] / params_dict['C_myo']

y0 = (params_dict['V_0'], params_dict['Na_i_0'], params_dict['K_i_0'], params_dict['Ca_i_0'], params_dict['H_i_0'],
      params_dict['Cl_i_0'], params_dict['a_ur_0'], params_dict['i_ur_0'], params_dict['vol_i_0'],
      params_dict['cal_0'])

male_OA_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
male_OA_VV, current_dict = Voltage_clamp(male_OA_solution)
male_OA_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

########################################################################################################################
# Female OA
params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
params_dict['g_Cl_b_bar'], params_dict['gBK'] = female_OA_scales

# params_dict['C_m'] = 6.3 * 37.93 / 21.56
# params_dict['NCX_scale'] = params_dict['C_m'] / params_dict['C_myo']
# params_dict['I_NaK_bar'] = params_dict['I_NaK_scale'] * 70.8253 * params_dict['C_m'] / params_dict['C_myo']

y0 = (params_dict['V_0'], params_dict['Na_i_0'], params_dict['K_i_0'], params_dict['Ca_i_0'], params_dict['H_i_0'],
      params_dict['Cl_i_0'], params_dict['a_ur_0'], params_dict['i_ur_0'], params_dict['vol_i_0'],
      params_dict['cal_0'])

female_OA_solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
female_OA_VV, current_dict = Voltage_clamp(female_OA_solution)
female_OA_currents = np.array([current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                            current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                            current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                            current_dict['I_BK']])

fig, axes = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(8, 3))

axes[0].plot(t, male_solution[:, 0], 'slategray', label='Male Control')
axes[0].plot(t, male_OA_solution[:, 0], 'tomato', label='Male OA')
# axes[0].plot(t, block_male_OA_solution[:, 0], 'royalblue', linestyle='dashed', label='OA (55% $I_{K\_DR}$ Block)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Voltage (mV)')
axes[0].legend()
axes[0].set_title('Male Control vs. OA')

axes[1].plot(t, female_solution[:, 0], 'slategray', label='Female Control')
axes[1].plot(t, female_OA_solution[:, 0], 'tomato', label='Female OA')
# axes[1].plot(t, block_female_OA_solution[:, 0], 'royalblue', linestyle='dashed', label='OA (65% $I_{K\_DR}$ Block)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Voltage (mV)')
axes[1].legend()
axes[1].set_title('Female Control vs. OA')

for ax in axes.flat:
    ax.yaxis.set_tick_params(labelbottom=True)
    # ax.set_ylim([-86, -46])

# plt.suptitle('Membrane Potential')

plt.tight_layout()
plt.savefig(figure_name, dpi=300)
print('Figure saved as:', figure_name)
plt.show()
