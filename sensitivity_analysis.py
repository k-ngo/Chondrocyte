import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate.odepack import odeint
import os
from chondrocyte import Voltage_clamp
import functions
from params import params_dict
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
import glob
import re

# Configs
dt = 0.1  # params_dict['dt']
female = False
control_input_folder = 'male'
OA_input_folder = 'male_OA'
figure_name = os.path.join('figures', 'sensitivity_analysis')

# Load datasets
control_detected_files = glob.glob(os.path.join(control_input_folder, '*.npy'))
control_all_voltage_and_ion_ss = np.load(control_detected_files[2])
control_all_parameters = np.load(control_detected_files[3])

OA_detected_files = glob.glob(os.path.join(OA_input_folder, '*.npy'))
OA_all_voltage_and_ion_ss = np.load(OA_detected_files[2])
OA_all_parameters = np.load(OA_detected_files[3])

# Get simulation time from file name
num_trials = int(re.search('%s(.*)%s' % ('__', 'trials_'), control_detected_files[0]).group(1))
t_final = int(re.search('%s(.*)%s' % ('trials_', 's_'), control_detected_files[0]).group(1))
t = np.linspace(0, t_final, int(t_final / dt))
figure_name += '_' + str(num_trials) + 'trials.png'

# Sensitivity analysis
# Performs linear regression analysis and plots the results
# all_ICs keys:
# 0: I_K_DR, 1: I_NaK, 2: I_NaCa, 3: I_Ca_ATP, 4: I_K_ATP, 5: I_K_2pore, 6: I_Na_b, 7: I_K_b, 8: I_Cl_b,
# 9: I_BK
# all_voltage_and_ions_ICs keys:
# 0: V, 1: Na_i, 2: K_i, 3: Ca_i, 4: H_i, 5: Cl_i, 6: a_ur, 7: i_ur, 8: vol_i, 9: cal
control_outputs = np.array([control_all_voltage_and_ion_ss[:, 0],
                            control_all_voltage_and_ion_ss[:, 1],
                            control_all_voltage_and_ion_ss[:, 2],
                            control_all_voltage_and_ion_ss[:, 3]])
OA_outputs = np.array([OA_all_voltage_and_ion_ss[:, 0],
                       OA_all_voltage_and_ion_ss[:, 1],
                       OA_all_voltage_and_ion_ss[:, 2],
                       OA_all_voltage_and_ion_ss[:, 3]])

# Perform the PLS regression and obtain the regression coefficients
# WT population
X = control_all_parameters  # predictor variables
Y = control_outputs.T  # response variable
pls2 = PLSRegression(n_components=2, max_iter=1000)
pls2.fit(X, Y)
cdf = pls2.coef_.T
# print(X, Y, sep='\n\n', end='\n\n')
# Diseased population
X = OA_all_parameters  # predictor variables
Y = OA_outputs.T  # response variable
OA_pls2 = PLSRegression(n_components=2, max_iter=1000)
OA_pls2.fit(X, Y)
OA_cdf = OA_pls2.coef_.T
# print(X, Y, sep='\n\n', end='\n\n')

# cdf = pd.concat([pd.DataFrame(X), pd.DataFrame(np.transpose(pls2.coef_))], axis=1).values.tolist()

output_names = ['$V_m$', '$[Na^{+}]_i$', '$[K^{+}]_i$', '$[Ca^{2+}]_i$']
output_units = ['mV', 'mM', 'mM', 'mM']
# current_labels = ['$I_{KDR}$', '$I_{NaK}$', '$I_{NaCa}$', '$I_{Ca-ATP}$', '$I_{K-ATP}$',
#                   '$I_{K-2pore}$', '$I_{Na-b}$', '$I_{K-b}$', '$I_{Cl-b}$', '$I_{K-Ca}$']
current_labels = ['$G_{KDR}$', '$G_{NaK}$', '$G_{NaCa}$', '$G_{Ca{-}ATP}$', '$G_{K{-}ATP}$',
                  '$G_{K{-}2pore}$', '$G_{Na{-}b}$', '$G_{K{-}b}$', '$G_{Cl{-}b}$', '$G_{K{-}Ca}$']
# current_labels = ['KDR', 'NaK', 'NaCa', 'Ca-ATP', 'K-ATP',
#                   'K-2pore', 'Na-b', 'K-b', 'Cl-b', 'K-Ca']


# Initialize graphs and plot results
x_loc = np.arange(len(current_labels))  # the label locations
width = 0.35  # the width of the bars

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
for i, ax in zip(range(len(output_names)), axes.flatten()):
    # print(len(labels), labels)
    # print(len(cdf[i]), cdf[i])

    if female:
        label = 'Female'
    else:
        label = 'Male'
    ax.bar(x_loc, cdf[i], width=-1. * width, align='edge', color='slategray', linewidth=0.5, edgecolor='k', label=label + ' Control')
    ax.bar(x_loc, OA_cdf[i], width=width, align='edge', color='tomato', linewidth=0.5, edgecolor='k', label=label + ' OA')

    # Set y-axis limits
    if output_names[i] == '$V_m$':
        ax.set_ylim([-4.1, 4.4])
    elif output_names[i] == '$[Na^{+}]_i$':
        ax.set_ylim([-14, 16])
    elif output_names[i] == '$[K^{+}]_i$':
        ax.set_ylim([-16, 12])
    elif output_names[i] == '$[Ca^{2+}]_i$':
        ax.set_ylim([-0.0009, 0.00065])

    ax.legend()
    ax.set_xticks(x_loc, current_labels)
    ax.set_title(output_names[i])
    ax.set_ylabel('Regression Coefficients')

plt.tight_layout()
plt.savefig(figure_name, dpi=300)
plt.show()
