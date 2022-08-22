import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate.odepack import odeint
from chondrocyte import Voltage_clamp
import functions
from params import params_dict

# Configs
num_trials = 100
sigma = 0.15  # Standard deviation for each parameter
t_final = 50000  # params_dict['t_final']  # 50000 ms
dt = 0.1  # params_dict['dt']
OA = False
female = True
figure_name = ''
t = np.linspace(0, t_final, int(t_final / dt))

# 1 - Generate parameters
# Generates random perturbations required for building a population of models

parameter_names = ['I_K_DR', 'I_NaK', 'I_NaCa', 'I_Ca_ATP', 'I_K_ATP', 'I_K_2pore', 'I_Na_b', 'I_K_b', 'I_Cl_b', 'I_BK']
all_parameters = []
diseased_all_parameters = []


# 2 - Obtain ICs
# Generates steady-state conditions for each model in the population
all_ICs = []
baseline_ICs = np.zeros((1, len(parameter_names)))
all_voltage_and_ions_ICs = []
baseline_voltage_and_ions_ICs = np.zeros((1, 10))
original_scales = params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
                  params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
                  params_dict['g_Cl_b_bar'], params_dict['gBK']

# Initialize plot for visualization purpose of steady-state conditions
fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(14, 10))

# Loop through each trial
trials_completed = 0
trials_attempted = 0
while trials_completed < num_trials:
    trials_attempted += 1
    print('Current trial is', trials_attempted)

    # Reset to original scales
    params_dict['g_K_DR'], params_dict['I_NaK_scale'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
    params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
    params_dict['g_Cl_b_bar'], params_dict['gBK'] = original_scales

    # Generate new scales
    scales = [np.random.lognormal(mean=np.log(params_dict['g_K_DR']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['I_NaK_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['NCX_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['I_Ca_ATP_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['sigma']), sigma=sigma),  # K_ATP
              np.random.lognormal(mean=np.log(params_dict['I_K_2pore_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['I_Na_b_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['g_K_b_bar']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['g_Cl_b_bar']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['gBK']), sigma=sigma)]

    # Check if I_NaK_scale is within specified scale
    # I_NaK_scale_limits = [0.1, 0.13]
    # while not I_NaK_scale_limits[0] <= scales[1] <= I_NaK_scale_limits[1]:
    #     scales[1] = np.random.lognormal(mean=np.log(np.mean(I_NaK_scale_limits)), sigma=sigma)
    scales[1] = 0.2

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

    # Change current scaling values/conductance to perturbed values
    params_dict['g_K_DR'], params_dict['I_NaK_scale'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
    params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
    params_dict['g_Cl_b_bar'], params_dict['gBK'] = scales

    params_dict['I_NaK_bar'] = params_dict['I_NaK_scale'] * 70.8253 * params_dict['C_m'] / params_dict['C_myo']

    if OA:
        params_dict['Q_10'] = 1.3
        params_dict['H_o'] = 10**(-7.4)

        params_dict['I_NaK_bar'] *= 2.2
        params_dict['I_Ca_ATP_scale'] *= 2
        params_dict['I_K_2pore_scale'] *= 0.2
        params_dict['g_K_DR'] *= 8.3
        params_dict['Q_10'] *= 3 / 1.3

    if female:
        # Female (Epi)
        params_dict['g_K_DR'] *= 0.96 / 1.20
        params_dict['NCX_scale'] *= 1.07 / 1.10
        params_dict['I_NaK_bar'] *= 0.87 / 0.92
        params_dict['g_K_b_bar'] *= 0.42 / 0.82

        # Female (Endo)
        # params_dict['g_K_DR'] *= 0.87 / 1.10
        # params_dict['NCX_scale'] *= 0.99 / 1.01
        # params_dict['I_NaK_bar'] *= 1 / 1
        # params_dict['g_K_b_bar'] *= 0.68 / 1.25

        # Female (Mid - Myocardium)
        # params_dict['g_K_DR'] *= 0.70 / 0.88
        # params_dict['NCX_scale'] *= 1.38 / 1.41
        # params_dict['I_NaK_bar'] *= 1 / 1
        # params_dict['g_K_b_bar'] *= 0.68 / 1.25

    # params_dict['H_o'] *= 10 ** (-6) / 10 ** (-7.4)

    # Generate ICs for each model in the population
    y0 = (V_0, Na_i_0, K_i_0, Ca_i_0, H_i_0, Cl_i_0, a_ur_0, i_ur_0, vol_i_0, cal_0)
    try:
        solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
        VV, current_dict = Voltage_clamp(solution)
    except:
        print('>> Trial skipped. Encountering error.')
        continue

    if solution[-1, 2] == 0:
        print('>> Trial skipped. Encountering error.')
        continue

    currents_ss = current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                  current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                  current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                  current_dict['I_BK']

    all_ICs.append(list(currents_ss))
    all_voltage_and_ions_ICs.append(solution[-1, :].tolist())

    for ax, label, values in zip(axes.flat[:10], solution_names, solution.T):
        # print(label, t[-2:], values[-2:])
        ax.plot(t, values)
        ax.set_xlabel('Time (s)')
        # solution_names = ['V', 'Na_i', 'K_i', 'Ca_i', 'H_i', 'Cl_i', 'a_ur', 'i_ur', 'vol_i', 'cal']
        if 'i' in label:
            ax.set_ylabel('Concentration (mM)')
        elif 'V' in label:
            ax.set_ylim([-111, -11])
            ax.set_ylabel('Voltage (mV)')
        else:
            ax.set_ylabel('Values')
        ax.set_title(label)
    # Plot currents for each trial
    for ax, label, values in zip(axes.flat[10:], parameter_names, currents_ss):
        ax.plot(VV, values)
        # sns.lineplot(x=VV, y=values, ax=ax)
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Current (pA/pF)')
        ax.set_title(label)

    trials_completed += 1

    if trials_completed == num_trials:
        break

all_voltage_and_ions_ICs = np.asarray(all_voltage_and_ions_ICs)

# Output steady state values
print('Steady state values:')
print(all_voltage_and_ions_ICs)
V_ss = all_voltage_and_ions_ICs[:, 0]
mean_V_ss = np.mean(V_ss)
std_V_ss = np.std(V_ss)

print('RMP :', V_ss)
print('mean:', mean_V_ss)
print('std :', std_V_ss)

plt.tight_layout()
print('>> Completed', trials_completed, 'trials.')

if female:
    figure_name += 'female_'
else:
    figure_name += 'male_'
if OA:
    figure_name += 'OA_'
figure_name += str(num_trials) + 'trials_'
figure_name += str(t_final) + 's_'
figure_name += str(round(mean_V_ss, 2)) + 'mean_'
figure_name += str(round(std_V_ss, 2)) + 'std'

plt.savefig(figure_name + '.png', dpi=300)
print('Figure saved as:', figure_name + '.png')
# plt.show()
