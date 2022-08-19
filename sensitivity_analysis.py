import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate.odepack import odeint
import os
from chondrocyte import Voltage_clamp
import functions
from params import params_dict
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

def single_voltage_clamp(solution):
    # get steady-state ion concentrations and voltage
    Ca_i_ss = solution[:, 3][solution.shape[0] - 1]
    Na_i_ss = solution[:, 1][solution.shape[0] - 1]
    K_i_ss = solution[:, 2][solution.shape[0] - 1]
    VV = solution[:, 0][-1]

    # create dictionary for saving currents
    current_dict = {
        'I_K_DR': 0,
        'I_NaK': 0,
        'I_NaCa': 0,
        'I_Ca_ATP': 0,
        'I_K_ATP': 0,
        'I_K_2pore': 0,
        'I_Na_b': 0,
        'I_K_b': 0,
        'I_Cl_b': 0,
        'I_leak': 0,
        'I_bq': 0,
        'I_BK': 0,
        'I_TRPV4': 0,
        'I_RMP': 0,
        'I_total': 0
    }

    # declear parameters
    C_m = params_dict['C_m'];
    K_o = params_dict['K_o'];
    Na_i_clamp = params_dict['Na_i_clamp']
    Ca_i_0 = params_dict['Ca_i_0'];
    K_i_0 = params_dict['K_i_0'];
    Q = params_dict['Q']
    E_Na = params_dict['E_Na'];
    g_K_b_bar = params_dict['g_K_b_bar'];
    temp = params_dict['temp']
    K_i = params_dict['K_i_0']

    # I_K_DR (printed in pA/pF)
    current_dict['I_K_DR'] = functions.DelayedRectifierPotassium(V=VV, enable_I_K_DR=True) / C_m

    # I_Na_K (pA; printed IV pA/pF)
    current_dict['I_NaK'] = functions.sodiumPotassiumPump(V=VV, K_o=K_o, Na_i_0=Na_i_clamp,
                                                          enable_I_NaK=True) / C_m

    # I_NaCa (pA; printed IV pA/pF)
    current_dict['I_NaCa'] = functions.sodiumCalciumExchanger(V=VV, Ca_i=Ca_i_0, Na_i_0=Na_i_clamp,
                                                              enable_I_NaCa=True) / C_m

    # I_Ca_ATP (pA)
    current_dict['I_Ca_ATP'] = functions.calciumPump(Ca_i=Ca_i_ss, enable_I_Ca_ATP=True)

    # I_K_ATP (pA?) Zhou/Ferrero, Biophys J, 2009
    current_dict['I_K_ATP'] = functions.potassiumPump(V=VV, K_i=K_i, K_o=K_o, E_K=-94.02, Na_i=Na_i_ss,
                                                      temp=temp, enable_I_K_ATP=True)

    # I_K_2pore (pA; pA/pF in print)
    # modeled as a simple Boltzmann relationship via GHK, scaled to match isotonic K+ data from Bob Clark
    current_dict['I_K_2pore'] = functions.twoPorePotassium(V=VV, K_i_0=K_i_0, K_o=K_o, Q=Q,
                                                           enable_I_K_2pore=True) / C_m

    # I_Na_b (pA; pA/pF in print)
    current_dict['I_Na_b'] = functions.backgroundSodium(V=VV, Na_i=None, E_Na=E_Na, enable_I_Na_b=True) / C_m

    # I_K_b (pA; pA/pF in print)
    current_dict['I_K_b'] = functions.backgroundPotassium(V=VV, K_i=None, K_o=None, g_K_b_bar=g_K_b_bar,
                                                          E_K=-83, enable_I_K_b=True) / C_m

    # I_Cl_b (pA; pA/pF in print)
    current_dict['I_Cl_b'] = functions.backgroundChloride(V=VV, Cl_i=None, enable_I_Cl_b=True) / C_m

    # I_leak (pA); not printed, added to I_bg
    current_dict['I_leak'] = functions.backgroundLeak(V=VV, enable_I_leak=False)

    # I_bg (pA; pA/pF in print)
    current_dict['I_bq'] = current_dict['I_Na_b'] + current_dict['I_K_b'] + current_dict['I_Cl_b'] + \
                           current_dict['I_leak']

    # I_K_Ca_act (new version) (pA; pA/pF in print), with converted Ca_i units for model
    current_dict['I_BK'] = functions.calciumActivatedPotassium(V=VV, Ca_i=Ca_i_0,
                                                               enable_I_K_Ca_act=True) / C_m

    # I TRPV4 (pA; pA/pF in print)
    current_dict['I_TRPV4'] = functions.TripCurrent(V=VV, enable_I_TRPV4=True) / C_m

    # I_RMP (pA; pA/pF in print)
    current_dict['I_RMP'] = current_dict['I_bq'] + current_dict['I_BK'] + current_dict['I_K_DR'] \
                            + current_dict['I_NaCa'] + current_dict['I_NaK'] + current_dict['I_K_2pore']

    # I_total (pA)
    current_dict['I_total'] = current_dict['I_NaK'] * C_m + current_dict['I_NaCa'] * C_m + \
                              current_dict['I_Ca_ATP'] + \
                              current_dict['I_K_DR'] * C_m + current_dict['I_K_2pore'] * C_m + \
                              current_dict['I_K_ATP'] + \
                              current_dict['I_BK'] + current_dict['I_Na_b'] * C_m + current_dict['I_K_b'] * C_m + \
                              current_dict['I_Cl_b'] * C_m + current_dict['I_leak'] + \
                              current_dict['I_TRPV4'] * C_m

    # slope_G = (current_dict['I_bq'][-1]-current_dict['I_bq'][0])*C_m/(VV[-1]-VV[0]) # pA/mV = nS
    # R = 1/slope_G # = GOhms

    return current_dict


# Configs
num_trials = 1000
sigma = 0.15  # Standard deviation for each parameter
t_final = 50000  # params_dict['t_final']  # 50000 ms
dt = 0.1  # params_dict['dt']
figure_name = 'reg_coeff_male_population_' + str(num_trials) + 'trials_' + str(t_final) + 's.png'
t = np.linspace(0, t_final, int(t_final / dt))
female = False

# 1 - Generate parameters
# Generates random perturbations required for building a population of models
# ['I_K_DR', 'I_NaK', 'I_NaCa', 'I_Ca_ATP', 'I_K_ATP', 'I_K_2pore', 'I_Na_b', 'I_K_b', 'I_Cl_b', 'I_BK']
#    0        1         2         3           4           5           6         7         8         9

parameter_names = ['I_K_DR', 'I_NaK', 'I_NaCa', 'I_Ca_ATP', 'I_K_ATP', 'I_K_2pore', 'I_Na_b', 'I_K_b', 'I_Cl_b', 'I_BK']
all_parameters = []
diseased_all_parameters = []

# 2 - Obtain ICs
# Generates steady-state conditions for each model in the population
baseline_ICs = np.zeros((1, len(parameter_names)))
baseline_voltage_and_ions_ICs = np.zeros((1, 10))
all_ICs = []
all_voltage_and_ions_ICs = []
diseased_all_ICs = []
diseased_all_voltage_and_ions_ICs = []
original_scales = params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
                  params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
                  params_dict['g_Cl_b_bar'], params_dict['gBK']

# Loop through each trial
trials_completed = 0
trials_attempted = 0
while trials_completed < num_trials:
    trials_attempted += 1
    print('Current trial is', trials_attempted)

    # Reset to original scales
    params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
    params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
    params_dict['g_Cl_b_bar'], params_dict['gBK'] = original_scales

    # Generate new scales
    scales = [np.random.lognormal(mean=np.log(params_dict['g_K_DR']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['I_NaK_bar']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['NCX_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['I_Ca_ATP_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['sigma']), sigma=sigma),  # K_ATP
              np.random.lognormal(mean=np.log(params_dict['I_K_2pore_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['I_Na_b_scale']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['g_K_b_bar']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['g_Cl_b_bar']), sigma=sigma),
              np.random.lognormal(mean=np.log(params_dict['gBK']), sigma=sigma)]

    if female:
        scales[0] *= 0.96 / 1.20  # g_K_DR
        scales[2] *= 1.07 / 1.10  # NCX_scale
        scales[1] *= 0.87 / 0.92  # I_NaK_bar
        scales[7] *= 0.42 / 0.82  # g_K_b_bar

    diseased_scales = scales.copy()
    diseased_scales[1] *= 2.2  # I_NaK_bar
    diseased_scales[3] *= 2  # I_Ca_ATP_scale
    diseased_scales[5] *= 0.2  # I_K_2pore_scale
    diseased_scales[0] *= 8.3  # g_K_DR
    diseased_scales[4] *= 3 ** ((23 - 36) / 10) / 1.3 ** ((23 - 36) / 10)  # I_K_ATP, this is Q_10 changes integrated into sigma

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

    params_dict['clamp_Na_i'] = False
    params_dict['clamp_K_i'] = False
    params_dict['calmp_Ca_i'] = False
    # params_dict['clamp_Cl_i'] = True

    parameter_names = ['I_K_DR', 'I_NaK', 'I_NaCa', 'I_Ca_ATP', 'I_K_ATP', 'I_K_2pore', 'I_Na_b', 'I_K_b', 'I_Cl_b',
                       'I_BK']
    solution_names = ['V', 'Na_i', 'K_i', 'Ca_i', 'H_i', 'Cl_i', 'a_ur', 'i_ur', 'vol_i', 'cal']

    # Obtain baseline results
    if trials_attempted == 0:
        y0 = (V_0, Na_i_0, K_i_0, Ca_i_0, H_i_0, Cl_i_0, a_ur_0, i_ur_0, vol_i_0, cal_0)
        solution = odeint(functions.rhs, y0, t, args=(params_dict,))
        current_dict = single_voltage_clamp(solution)
        currents_ss = current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
                      current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
                      current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
                      current_dict['I_BK']
        # Populate baseline ICs with steady-state currents, V, and ion concentrations
        # baseline_ICs[0, :] = list(currents) + solution[-1, :].tolist() \
        # Populate baseline ICs with ONLY steady-state currents
        baseline_ICs = list(currents_ss)
        baseline_voltage_and_ions_ICs = solution[-1, :].tolist()

    ####################################################################################################################
    # Generate population for diseased state

    # Change current scaling values/conductance to perturbed values
    # Note: sigma is for I_K_ATP

    params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
    params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
    params_dict['g_Cl_b_bar'], params_dict['gBK'] = diseased_scales

    y0 = (V_0, Na_i_0, K_i_0, Ca_i_0, H_i_0, Cl_i_0, a_ur_0, i_ur_0, vol_i_0, cal_0)
    try:
        solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
        current_dict = single_voltage_clamp(solution)
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

    # Populate population ICs with steady-state currents, V, and ion concentrations
    # all_ICs[i, :] = list(currents) + solution[-1, :].tolist()
    # Populate population ICs with ONLY steady-state currents
    diseased_all_ICs.append(list(currents_ss))
    diseased_all_voltage_and_ions_ICs.append(solution[-1, :].tolist())

    ####################################################################################################################
    # Generate baseline population
    params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
    params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
    params_dict['g_Cl_b_bar'], params_dict['gBK'] = scales

    y0 = (V_0, Na_i_0, K_i_0, Ca_i_0, H_i_0, Cl_i_0, a_ur_0, i_ur_0, vol_i_0, cal_0)
    try:
        solution = odeint(functions.rhs, y0, t, args=(params_dict,))
        current_dict = single_voltage_clamp(solution)
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
    # Populate population ICs with steady-state currents, V, and ion concentrations
    # all_ICs[i, :] = list(currents) + solution[-1, :].tolist()
    # Populate population ICs with ONLY steady-state currents
    all_ICs.append(list(currents_ss))
    all_voltage_and_ions_ICs.append(solution[-1, :].tolist())

    # Add data to result arrays
    all_parameters.append(scales)
    diseased_all_parameters.append(diseased_scales)
    trials_completed += 1

    if trials_completed == num_trials:
        break

# 3 - Sensitivity analysis
# Performs linear regression analysis and plots the results
# all_ICs keys:
# 0: I_K_DR, 1: I_NaK, 2: I_NaCa, 3: I_Ca_ATP, 4: I_K_ATP, 5: I_K_2pore, 6: I_Na_b, 7: I_K_b, 8: I_Cl_b,
# 9: I_BK
# all_voltage_and_ions_ICs keys:
# 0: V, 1: Na_i, 2: K_i, 3: Ca_i, 4: H_i, 5: Cl_i, 6: a_ur, 7: i_ur, 8: vol_i, 9: cal

# Restrict analysis to a subset of population parameters
output_names = ['$V_m$', '$[Na^{+}]_i$', '$[K^{+}]_i$', '$[Ca^{2+}]_i$']
output_units = ['mV', 'mM', 'mM', 'mM']
N_outputs = len(output_names)

all_parameters = np.asarray(all_parameters)
diseased_all_parameters = np.asarray(diseased_all_parameters)

baseline_voltage_and_ions_ICs = np.asarray(baseline_voltage_and_ions_ICs)
all_voltage_and_ions_ICs = np.asarray(all_voltage_and_ions_ICs)
diseased_all_voltage_and_ions_ICs = np.asarray(diseased_all_voltage_and_ions_ICs)

baseline_outputs = np.array([baseline_voltage_and_ions_ICs[0, 0],
                             baseline_voltage_and_ions_ICs[0, 1],
                             baseline_voltage_and_ions_ICs[0, 2],
                             baseline_voltage_and_ions_ICs[0, 3]])
all_outputs = np.array([all_voltage_and_ions_ICs[:, 0],
                        all_voltage_and_ions_ICs[:, 1],
                        all_voltage_and_ions_ICs[:, 2],
                        all_voltage_and_ions_ICs[:, 3]])
diseased_all_outputs = np.array([diseased_all_voltage_and_ions_ICs[:, 0],
                                 diseased_all_voltage_and_ions_ICs[:, 1],
                                 diseased_all_voltage_and_ions_ICs[:, 2],
                                 diseased_all_voltage_and_ions_ICs[:, 3]])

# Perform the PLS regression and obtain the regression coefficients
# WT population
X = all_parameters  # predictor variables
Y = all_outputs.T  # response variable
pls2 = PLSRegression(n_components=2, max_iter=1000)
pls2.fit(X, Y)
cdf = pls2.coef_.T
# print(X, Y, sep='\n\n', end='\n\n')
# Diseased population
X = diseased_all_parameters  # predictor variables
Y = diseased_all_outputs.T  # response variable
diseased_pls2 = PLSRegression(n_components=2, max_iter=1000)
diseased_pls2.fit(X, Y)
diseased_cdf = diseased_pls2.coef_.T
# print(X, Y, sep='\n\n', end='\n\n')

# cdf = pd.concat([pd.DataFrame(X), pd.DataFrame(np.transpose(pls2.coef_))], axis=1).values.tolist()

labels = ['$I_{K\_DR}$', '$I_{NaK}$', '$I_{NaCa}$', '$I_{Ca\_ATP}$', '$I_{K\_ATP}$', '$I_{K\_2pore}$', '$I_{Na\_b}$', '$I_{K\_b}$', '$I_{Cl\_b}$',
          '$I_{BK}$']  # , 'V', 'Na_i', 'K_i', 'Ca_i', 'H_i', 'Cl_i', 'a_ur', 'i_ur', 'vol_i', 'cal']

# Initialize and plot results

x_loc = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
for i, ax in zip(range(len(output_names)), axes.flatten()):
    # print(len(labels), labels)
    # print(len(cdf[i]), cdf[i])

    ax.bar(x_loc, cdf[i], width=-1. * width, align='edge', color='slategray', label='Control')
    ax.bar(x_loc, diseased_cdf[i], width=width, align='edge', color='tomato', label='OA')

    ax.legend()
    ax.set_xticks(x_loc, labels)
    ax.set_title(output_names[i])
    ax.set_ylabel('Regression Coefficients')

plt.tight_layout()
plt.savefig(figure_name, dpi=300)
plt.show()
