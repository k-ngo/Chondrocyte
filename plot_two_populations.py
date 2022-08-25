import numpy as np
import matplotlib.pyplot as plt
from params import params_dict
import os
import glob
import re

# Configs
dt = 0.1  # params_dict['dt']
first_input_folder = '100male'
second_input_folder = '100female'
figure_name = os.path.join('figures', 'two_populations.png')

# Load datasets
first_detected_files = sorted(glob.glob(os.path.join(first_input_folder, '*.npy')))
first_all_currents = np.load(first_detected_files[0])
first_all_parameters = np.load(first_detected_files[1])
first_all_voltage_and_ion = np.load(first_detected_files[2])
first_all_voltage_and_ion_ss = np.load(first_detected_files[3])

second_detected_files = sorted(glob.glob(os.path.join(second_input_folder, '*.npy')))
second_all_currents = np.load(second_detected_files[0])
second_all_parameters = np.load(second_detected_files[1])
second_all_voltage_and_ion = np.load(second_detected_files[2])
second_all_voltage_and_ion_ss = np.load(second_detected_files[3])

# Get simulation time from file name
num_trials = int(re.search('%s(.*)%s' % ('__', 'trials_'), first_detected_files[0]).group(1))
t_final = int(re.search('%s(.*)%s' % ('trials_', 's_'), first_detected_files[0]).group(1))
t = np.linspace(0, t_final, int(t_final / dt))

# Initialize plot for visualization purpose of steady-state conditions
fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(14, 10))
current_labels = ['$I_{KDR}$', '$I_{NaK}$', '$I_{NaCa}$', '$I_{Ca-ATP}$', '$I_{K-ATP}$', '$I_{K-2pore}$', '$I_{Na-b}$', '$I_{K-b}$', '$I_{Cl-b}$', '$I_{K-Ca}$']
voltage_and_ion_labels = ['$V_m$', '$[Na^{+}]_i$', '$[K^{+}]_i$', '$[Ca^{2+}]_i$', '$[H^{+}]_{i}$', '$[Cl^{-}]_{i}$', '$a_{ur}$', '$i_{ur}$', '$vol_{i}$', '$cal$']

# Plot each model in the population
for model in first_all_voltage_and_ion:
    for ax, label, values in zip(axes.flat[:10], voltage_and_ion_labels, model):
        # Detect value of stride used when saving data ("stride" variable from generate_population.py)
        stride = int(len(t) / len(values))
        ax.plot(t[::stride], values, color='tomato', alpha=0.2)
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        if 'i' in label and label != '$vol_{i}$':
            ax.set_ylabel('Concentration (mM)')
        elif 'V' in label:
            ax.set_ylabel('Voltage (mV)')
        else:
            ax.set_ylabel('Arbitrary Units')

for model in second_all_voltage_and_ion:
    for ax, label, values in zip(axes.flat[:10], voltage_and_ion_labels, model):
        # Detect value of stride used when saving data ("stride" variable from generate_population.py)
        stride = int(len(t) / len(values))
        ax.plot(t[::stride], values, color='dodgerblue', alpha=0.2)
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        if 'i' in label and label != '$vol_{i}$':
            ax.set_ylabel('Concentration (mM)')
        elif 'V' in label:
            ax.set_ylabel('Voltage (mV)')
        else:
            ax.set_ylabel('Arbitrary Units')

# VV = np.linspace(params_dict['V_start'], params_dict['V_end'], params_dict['V_step_size'])
# for model in all_currents:
#     for ax, label, values in zip(axes.flat[10:], current_labels, model):
#         ax.plot(VV[::stride], values)
#         ax.set_xlabel('Voltage (mV)')
#         ax.set_ylabel('Current (pA/pF)')
#         ax.set_title(label)

plt.tight_layout()
plt.savefig(figure_name, dpi=300)
print('Figure saved as:', figure_name)
