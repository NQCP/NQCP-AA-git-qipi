from pathlib import Path

from pygridsynth.gridsynth import gridsynth_gates
from pygridsynth.myplot import plot_sol
import mpmath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

plt.style.use(_REPO_ROOT / 'plotstylefile.mplstyle')
plotdir = str(_REPO_ROOT / "paper_plots") + "/"

# generate 1000 numbers between 0 and 2*pi
angles = np.linspace(0, 2*np.pi, 1000)

precisions = [1e-1, 1e-2, 1e-3, 1e-4]
mpmath.mp.dps = 128

gate_decomposition_list = []
T_count_array = np.zeros((len(precisions), len(angles)))

for precision in precisions:
    epsilon = mpmath.mpmathify(str(precision))
    T_count_list = []
    for angle in angles:
        theta = mpmath.mpmathify(str(angle))
        gates = gridsynth_gates(theta=theta, epsilon=epsilon)
        gate_decomposition_list.append(gates)

        # t count
        t_count = 0
        for gate in gates:
            if gate == 'T':
                t_count += 1

        # s count
        # s_count = 0
        # for gate in gates:
        #     if gate == 'S':
        #         s_count += 1

        # h count 
        # h_count = 0
        # for gate in gates:
        #     if gate == 'H':
        #         h_count += 1
                
        
        T_count_list.append(t_count)
    T_count_array[precisions.index(precision)] = T_count_list


# plot the T count vs angle for each precision (colors from plotstylefile prop_cycle)
plt.figure(figsize=(11, 7.5))
for i in range(len(precisions)):
    std = np.std(T_count_array[i][T_count_array[i] != 0])
    mean = np.mean(T_count_array[i][T_count_array[i] != 0])

    exponent = int(np.log10(precisions[i]))
    plt.plot(angles, T_count_array[i], label=rf"$10^{{{exponent}}}$")
    plt.axhline(y=mean, color='black', linestyle='--')
    plt.fill_between(angles, mean - std, mean + std, alpha=0.2)

# plt.title("T count vs Angle for different precisions")
fig = plt.gcf()
ax = plt.gca()
# Set x-axis ticks to only show 0, π, and 2π
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
_rc = plt.rcParams
ax.set_xlabel(r'Angle $\phi$', fontsize=_rc['axes.labelsize'])
ax.set_ylabel(
    r'$\mathrm{T}$ gate count per $R_z(\phi)$', fontsize=_rc['axes.labelsize']
)
ax.tick_params(axis='x', which='major', labelsize=_rc['xtick.labelsize'])
ax.tick_params(axis='y', which='major', labelsize=_rc['ytick.labelsize'])

handles, labels = ax.get_legend_handles_labels()
mean_handle = mlines.Line2D(
    [0], [0], color='black', linestyle='--', linewidth=1.0, label='mean'
)
handles.append(mean_handle)
labels.append('mean')
# grey shaded box representing the standard deviation band (matches fill_between alpha)
std_handle = mpatches.Patch(facecolor='grey', alpha=0.2, label='std')
handles.append(std_handle)
labels.append('std')
# matplotlib fills ncol legends column-by-column; reorder for row-wise layout
legend_ncol = 3
legend_handles = list(handles)
legend_labels = list(labels)
n_leg = len(legend_handles)
legend_nrow = int(np.ceil(n_leg / legend_ncol))
_leg_grid_h = [[None] * legend_ncol for _ in range(legend_nrow)]
_leg_grid_l = [[None] * legend_ncol for _ in range(legend_nrow)]
for idx in range(n_leg):
    r, c = idx // legend_ncol, idx % legend_ncol
    _leg_grid_h[r][c] = legend_handles[idx]
    _leg_grid_l[r][c] = legend_labels[idx]
legend_handles_rowwise = []
legend_labels_rowwise = []
for c in range(legend_ncol):
    for r in range(legend_nrow):
        if _leg_grid_h[r][c] is not None:
            legend_handles_rowwise.append(_leg_grid_h[r][c])
            legend_labels_rowwise.append(_leg_grid_l[r][c])
fig.legend(
    legend_handles_rowwise,
    legend_labels_rowwise,
    loc='center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=legend_ncol,
    title=r'Precision $\epsilon$',
    fontsize=_rc['legend.fontsize'],
    title_fontsize=_rc['legend.title_fontsize'],
)
# plt.tight_layout(pad=2.0)
plt.savefig(plotdir + 'figure9.pdf', dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()

