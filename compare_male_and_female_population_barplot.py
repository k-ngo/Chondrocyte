import matplotlib.pyplot as plt

mean_validation_control = -69.11  # -21.2
mean_validation_OA = -69.11 * -26 / -39  # -23

std_validation_control = 4
std_validation_OA = 4

mean_male_control = -69.11
std_male_control = 4.71

mean_male_OA = -53.87
std_male_OA = 3.21

mean_female_control = -58.21
std_female_control = 5.75

mean_female_OA = -49.03
std_female_OA = 1.34

labels = ['Scaled Male\nExp. Validation', 'Male', 'Female']

control = [mean_validation_control, mean_male_control, mean_female_control]
OA = [mean_validation_OA, mean_male_OA, mean_female_OA]
control_error = [std_validation_control, std_male_control, std_female_control]
OA_error = [std_validation_OA, std_male_OA, std_female_OA]

x_loc = range(3)
width = 0.35  # the width of the bars

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

ax.bar(x_loc, control, yerr=control_error, width=-1 * width, align='edge', color='slategray', linewidth=0.5, edgecolor='k', label='Control')
ax.bar(x_loc, OA, yerr=OA_error, width=width, align='edge', color='tomato', linewidth=0.5, edgecolor='k', label='OA')

ax.legend(ncol=2)
ax.set_xticks(x_loc, labels)
ax.set_title('Resting Membrane Potential in Control vs. OA')
ax.set_ylabel('Voltage (mV)')
ax.set_ylim([-89, 0])

plt.tight_layout()
plt.savefig('figures/change_in_RMP.png', dpi=300)
plt.show()
