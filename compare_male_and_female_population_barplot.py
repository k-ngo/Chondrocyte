import matplotlib.pyplot as plt

mean_validation_control = -43.85  # -21.2
mean_validation_OA = -43.85 * -23 / -21.2  # -23

std_validation_control = 0.923
std_validation_OA = 0.837

mean_male_control = -43.85
std_male_control = 3.15

mean_male_OA = -45.64
std_male_OA = 1.22

mean_female_control = -42.48
std_female_control = 3.42

mean_female_OA = -44.51
std_female_OA = 1.25

labels = ['Scaled Male\nExp. Validation', 'Male', 'Female']

control = [mean_validation_control, mean_male_control, mean_female_control]
OA = [mean_validation_OA, mean_male_OA, mean_female_OA]
control_error = [std_validation_control, std_male_control, std_female_control]
OA_error = [std_validation_OA, std_male_OA, std_female_OA]

x_loc = range(3)
width = 0.35  # the width of the bars

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

ax.bar(x_loc, control, yerr=control_error, width=-1 * width, align='edge', color='slategray', label='Control')
ax.bar(x_loc, OA, yerr=OA_error, width=width, align='edge', color='tomato', label='OA')

ax.legend(ncol=2)
ax.set_xticks(x_loc, labels)
ax.set_title('Resting Membrane Potential in Control vs. OA')
ax.set_ylabel('Voltage (mV)')
ax.set_ylim([-59, 0])

plt.tight_layout()
plt.savefig('change_in_RMP.png', dpi=300)
plt.show()
