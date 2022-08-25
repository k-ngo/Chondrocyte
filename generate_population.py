import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate.odepack import odeint
from chondrocyte import Voltage_clamp
import functions
from params import params_dict
import os
import glob

# Configs
num_trials = 10
sigma = 0.15  # Standard deviation for each parameter
t_final = 50000  # params_dict['t_final']  # 50000 ms
dt = 0.1  # params_dict['dt']
OA = False
female = True
output_folder = 'female'
output_name = ''
t = np.linspace(0, t_final, int(t_final / dt))

# Advanced configs
stride = 10  # when saving data, skip this many data points to speed up saving and free up memory usage

# Initialize arrays and location for storage
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
parameter_names = ['I_K_DR', 'I_NaK', 'I_NaCa', 'I_Ca_ATP', 'I_K_ATP', 'I_K_2pore', 'I_Na_b', 'I_K_b', 'I_Cl_b', 'I_BK']
all_parameters = []
all_currents = []
all_voltage_and_ion = []
all_voltage_and_ion_ss = []

# Set custom scales for different conditions
params_dict['I_NaK_scale'] = 0.35
params_dict['I_NaK_bar'] = params_dict['I_NaK_scale']*70.8253*params_dict['C_m']/params_dict['C_myo']

baseline_scales = [params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
                   params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
                   params_dict['g_Cl_b_bar'], params_dict['gBK']]

parameter_scales = baseline_scales.copy()

if female:
    parameter_scales[1] *= 0.87 / 0.92  # I_NaK_bar
    parameter_scales[0] *= 0.96 / 1.20  # g_K_DR
    parameter_scales[2] *= 1.07 / 1.10  # NCX_scale
    parameter_scales[7] *= 0.42 / 0.82  # g_K_b_bar

if OA:
    parameter_scales[1] *= 2.2  # I_NaK_bar
    parameter_scales[3] *= 2  # I_Ca_ATP_scale
    parameter_scales[5] *= 0.2  # I_K_2pore_scale
    parameter_scales[0] *= 8.3  # g_K_DR
    parameter_scales[4] *= 3 ** ((23 - 36) / 10) / 1.3 ** ((23 - 36) / 10)  # I_K_ATP, this is Q_10 changes integrated into sigma

# Set universal parameters
params_dict['clamp_Na_i'] = False
params_dict['clamp_K_i'] = False
params_dict['calmp_Ca_i'] = False
# params_dict['clamp_Cl_i'] = True
parameter_names = ['I_KDR', 'I_NaK', 'I_NaCa', 'I_Ca-ATP', 'I_K-ATP', 'I_K-2pore', 'I_Na-b', 'I_K-b', 'I_Cl-b',
                   'I_KCa']
solution_names = ['V', 'Na_i', 'K_i', 'Ca_i', 'H_i', 'Cl_i', 'a_ur', 'i_ur', 'vol_i', 'cal']

# Loop through each trial
trials_completed = 0
trials_attempted = 0
while trials_completed < num_trials:
    trials_attempted += 1
    print('Trials completed: ', trials_completed, '| Trials attempted: ', trials_attempted)

    # Reset to original scales
    params_dict['g_K_DR'], params_dict['I_NaK_bar'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
    params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
    params_dict['g_Cl_b_bar'], params_dict['gBK'] = parameter_scales

    # Generate new scales
    new_scales = [np.random.lognormal(mean=np.log(params_dict['g_K_DR']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['I_NaK_bar']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['NCX_scale']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['I_Ca_ATP_scale']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['sigma']), sigma=sigma),  # K_ATP
                  np.random.lognormal(mean=np.log(params_dict['I_K_2pore_scale']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['I_Na_b_scale']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['g_K_b_bar']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['g_Cl_b_bar']), sigma=sigma),
                  np.random.lognormal(mean=np.log(params_dict['gBK']), sigma=sigma)]

    # Change current scaling values/conductance to perturbed values
    params_dict['g_K_DR'], params_dict['I_NaK_scale'], params_dict['NCX_scale'], params_dict['I_Ca_ATP_scale'], \
    params_dict['sigma'], params_dict['I_K_2pore_scale'], params_dict['I_Na_b_scale'], params_dict['g_K_b_bar'], \
    params_dict['g_Cl_b_bar'], params_dict['gBK'] = new_scales

    # Obtain steady-state conditions
    y0 = (params_dict['V_0'], params_dict['Na_i_0'], params_dict['K_i_0'], params_dict['Ca_i_0'], params_dict['H_i_0'],
          params_dict['Cl_i_0'], params_dict['a_ur_0'], params_dict['i_ur_0'], params_dict['vol_i_0'],
          params_dict['cal_0'])

    try:
        solution = odeint(functions.rhs, y0, t, args=(params_dict,), mxstep=10000)
        VV, current_dict = Voltage_clamp(solution)
    except:
        print('>> Trial skipped. Encountering error!')
        continue

    if solution[-1, 2] == 0:
        print('>> Trial skipped. Encountering error!')
        continue

    currents = current_dict['I_K_DR'], current_dict['I_NaK'], current_dict['I_NaCa'], \
               current_dict['I_Ca_ATP'], current_dict['I_K_ATP'], current_dict['I_K_2pore'], \
               current_dict['I_Na_b'], current_dict['I_K_b'], current_dict['I_Cl_b'], \
               current_dict['I_BK']

    # Append to result arrays
    all_currents.append(list([i[::stride] for i in currents]))
    all_voltage_and_ion.append(list([i[::stride] for i in solution.T]))
    all_voltage_and_ion_ss.append(solution[-1, :].tolist())
    all_parameters.append(new_scales)

    trials_completed += 1
    if trials_completed == num_trials:
        break
print('>> Completed', trials_completed, 'trials')

# Transform results into numpy arrays
all_currents = np.asarray(all_currents)
all_voltage_and_ion = np.asarray(all_voltage_and_ion)
all_voltage_and_ion_ss = np.asarray(all_voltage_and_ion_ss)
all_parameters = np.asarray(all_parameters)

# Output steady-state values
print('>> Steady-state Resting Membrane Potential:')
V_ss = all_voltage_and_ion_ss[:, 0]
mean_V_ss = np.mean(V_ss)
std_V_ss = np.std(V_ss)
print('   Mean:', round(mean_V_ss, 2))
print('   STD :', round(std_V_ss, 2))

# Set names of output files
if female:
    output_name += 'female_'
else:
    output_name += 'male_'
if OA:
    output_name += 'OA_'
output_name += '_' + str(num_trials) + 'trials_'
output_name += str(t_final) + 's_'
output_name += str(round(mean_V_ss, 2)) + 'mean_'
output_name += str(round(std_V_ss, 2)) + 'std'

# Move old data to archive
if glob.glob(os.path.join(output_folder, '*.npy')):
    print('>> Archiving old data...')
    if not os.path.exists(os.path.join(output_folder, 'archive')):
        os.makedirs(os.path.join(output_folder, 'archive'))
    for file in glob.glob(os.path.join(output_folder, '*.npy')):
        os.replace(file, os.path.join(output_folder, 'archive', os.path.basename(file)))

# Save population data to files
print('>> Saving data to files')
np.save(os.path.join(output_folder, 'currents_' + output_name), all_currents)
np.save(os.path.join(output_folder, 'voltage_and_ion_' + output_name), all_voltage_and_ion)
np.save(os.path.join(output_folder, 'voltage_and_ion_ss_' + output_name), all_voltage_and_ion_ss)
np.save(os.path.join(output_folder, 'parameters_' + output_name), all_parameters)

print('>> Data saved in folder', output_folder)
print('   currents_' + output_name + '.npy')
print('   voltage_and_ion_' + output_name + '.npy')
print('   voltage_and_ion_ss_' + output_name + '.npy')
print('   parameters_' + output_name + '.npy')
