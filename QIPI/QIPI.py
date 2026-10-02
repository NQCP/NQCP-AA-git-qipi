"""
Standalone combined QIPI plot: H2 (row 1), LiH (row 2), BeH2 (row 3).

Data is loaded from QIPI/data/<molecule>.pkl, which is produced beforehand by
QIPI/generate_data.py (run that first; BeH2 is slow).
"""

import os
import sys
import pickle
import warnings

import numpy as np
import matplotlib

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_THIS_DIR, 'data')

plt.style.use(os.path.join(_REPO_ROOT, 'plotstylefile.mplstyle'))

# Gradient colormap for eigenvalue vertical lines
color1 = '#3929b7'  # blue
color2 = '#901a1e'  # red
n_bins = 256
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [color1, color2], N=n_bins)

# ----- Degree color mapping  -----
default_color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b',
                       '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

degree_color_map = {
    2: '#1f77b4',  # blue
    4: '#ff7f0e',  # orange
    8: '#2ca02c',  # green
}


def get_degree_colors(degrees_list):
    """Get colors for degrees, maintaining consistency across plots."""
    global degree_color_map
    new_degrees = [d for d in degrees_list if d not in degree_color_map]
    used_colors = set(degree_color_map.values())
    for degree in new_degrees:
        for color in default_color_cycle:
            if color not in used_colors:
                degree_color_map[degree] = color
                used_colors.add(color)
                break
        else:
            max_used_idx = max([default_color_cycle.index(c) for c in used_colors if c in default_color_cycle], default=-1)
            next_idx = (max_used_idx + 1) % len(default_color_cycle)
            degree_color_map[degree] = default_color_cycle[next_idx]
            used_colors.add(default_color_cycle[next_idx])
    return [degree_color_map[d] for d in degrees_list]


# ----- Parameters  -----
# Degrees are per-molecule (read from each data file's 'degrees'). legend_degrees
# is the union across all molecules, given as M (displayed d = 2*M):
#   H2 -> d=[4,8,16], LiH -> d=[8,16,32], BeH2 -> d=[16,32,64]
legend_degrees = [2, 4, 8, 16, 32]   # All degrees shown in legend (as M)
precisions = [0.1, 0.01, 0.001]
xlim = (-1, 1)

# Per-precision styling used only in rotation-decomposition mode 
linewidth_convergence = [2.5, 2, 2]
precision_linestyles = [':', (0, (3, 2, 1, 2, 1, 2)), '--']
precision_markers = ['s', '2', 'o']
precision_markersizes = [5, 20, 5]
alpha_transparency = [0.7, 0.7, 1]

ROT_SUFFIX = '_with_rot_decompose'

get_degree_colors(legend_degrees)


def load_data(molecule, use_rotation_decomposition=False):
    suffix = ROT_SUFFIX if use_rotation_decomposition else ''
    path = os.path.join(DATA_DIR, f'{molecule}{suffix}.pkl')
    if not os.path.exists(path):
        rot_arg = ' --rot' if use_rotation_decomposition else ''
        raise FileNotFoundError(
            f"Missing data file: {path}\nRun: python -m QIPI.generate_data{rot_arg} {molecule}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_polynomial(ax, d, use_rotation_decomposition=False):
    """Left panel: polynomial R_d(x, Delta), reproducing the original ax1."""
    rSeigenValues = d['rSeigenValues']
    targetIdx = d['targetIdx']
    delta_scaled = d['delta_scaled']
    x = d['x']
    first_element_array = d['first_element_array']
    degrees = d['degrees']

    colors = custom_cmap(np.linspace(0, 1, int(len(rSeigenValues) * (1.2))))

    # Non-target eigenvalue line style: H2 keeps the original (lw=1, alpha=0.5);
    # the denser LiH / BeH2 spectra use thinner, fainter lines.
    if d['molecule'] == 'H2':
        other_lw, other_alpha = 1, 0.5
    else:
        other_lw, other_alpha = 0.6, 0.1

    # Eigenvalue vertical lines
    for i in range(0, len(rSeigenValues)):
        color = colors[i]
        if i == targetIdx:
            ax.axvline(rSeigenValues[i], linewidth=1.5, color=color)
        else:
            ax.axvline(rSeigenValues[i], linewidth=other_lw, linestyle=':', color=color, alpha=other_alpha)

    # Delta band + annotation
    ax.axvspan(-delta_scaled, delta_scaled, color='grey', alpha=0.4)
    ax.text(0, 1.02, r'$\leftarrow\Delta = ' + str(np.round(delta_scaled, 2)) + r'\rightarrow$',
            ha='center', va='bottom', transform=ax.get_xaxis_transform())

    # Zero line (only in normal QSVT mode)
    if not use_rotation_decomposition:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2)

    # Polynomial curves
    for degree_idx, degree in enumerate(degrees):
        for p_idx, precision in enumerate(precisions):
            if use_rotation_decomposition:
                ax.plot(x, first_element_array[degree_idx, p_idx, :],
                        color=degree_color_map[degree], linewidth=linewidth_convergence[p_idx],
                        alpha=alpha_transparency[p_idx], linestyle=precision_linestyles[p_idx])
            else:
                # Solid lines; precision does not change the curve here
                ax.plot(x, first_element_array[degree_idx, p_idx, :],
                        color=degree_color_map[degree], linewidth=1.2, alpha=0.9, linestyle='-')

    ax.set_xlim(d.get('xlim', xlim))
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$R_d(x,\Delta)$')


def plot_convergence(ax, d, use_rotation_decomposition=False):
    """Right panel: convergence overlap, reproducing the original ax2."""
    iterVec = d['iterVec']
    oveOfIter = d['oveOfIter']
    overlap_lists = d['overlap_lists']
    degrees = d['degrees']

    # Exact inverse (only in normal QSVT mode)
    if not use_rotation_decomposition:
        ax.plot(iterVec, np.abs(oveOfIter), '--', color='black', linewidth=1.2, label="Exact inverse")

    for degree_idx, M in enumerate(degrees):
        for precision_idx, precision in enumerate(precisions):
            overlap_list = overlap_lists[degree_idx][precision_idx]
            if use_rotation_decomposition:
                ax.plot(range(len(overlap_list)), overlap_list, color=degree_color_map[M],
                        linewidth=linewidth_convergence[precision_idx],
                        linestyle=precision_linestyles[precision_idx], marker=precision_markers[precision_idx],
                        markersize=precision_markersizes[precision_idx],
                        markeredgecolor='black', markeredgewidth=0.8, alpha=alpha_transparency[precision_idx])
            else:
                ax.plot(range(len(overlap_list)), overlap_list, color=degree_color_map[M], linewidth=1.2,
                        linestyle='-', marker='o', markersize=7,
                        markeredgecolor='black', markeredgewidth=0.8, alpha=0.9)

    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('Overlap ' + r'$|\langle \Psi_{t} | \Psi_k \rangle|^2$')
    ax.set_ylim(0, 1.1)


def build_shared_legends(fig, ref_data, use_rotation_decomposition=False):
    """One shared legend block below all rows. In normal QSVT mode: System
    (target eigenvalue + exact inverse) and Degree (d boxes). In
    rotation-decomposition mode: System (target eigenvalue), Precision (epsilon)
    and Degree. Uses the BeH2-style legend boxes."""
    rSeigenValues = ref_data['rSeigenValues']
    targetIdx = ref_data['targetIdx']
    colors = custom_cmap(np.linspace(0, 1, int(len(rSeigenValues) * (1.2))))
    target_color = colors[targetIdx]

    legends = []

    # ----- System legend: target eigenvalue, other eigenvalues
    #        (+ exact inverse in normal mode) -----
    target_line = plt.Line2D([0], [0], color=target_color, linewidth=1.5,
                             label='Target eigenvalue ' + r'$\tilde{\lambda}_t$')
    other_line = plt.Line2D([0], [0], color='black', linestyle=':', linewidth=0.6,
                            label='Other eigenvalues')
    sys_handles = [target_line, other_line]
    sys_labels = ['Target eigenvalue ' + r'$\tilde{\lambda}_t$', 'Other eigenvalues']
    if not use_rotation_decomposition:
        exact_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2,
                                label='Exact inverse')
        sys_handles.append(exact_line)
        sys_labels.append('Exact inverse')
    # In rotation-decomposition mode three legends share the row (eigenvalue |
    # precision | degree), so spread the eigenvalue legend further left to keep
    # the precision legend centred.
    sys_anchor_x = 0.18 if use_rotation_decomposition else 0.30
    leg1 = fig.legend(sys_handles, sys_labels,
                      loc='center', bbox_to_anchor=(sys_anchor_x, -0.02),
                      frameon=True, fancybox=True, shadow=True)
    legends.append(leg1)

    # ----- Precision legend (rotation-decomposition mode only) -----
    if use_rotation_decomposition:
        precision_handles = []
        precision_labels = []
        for p_idx, precision in enumerate(precisions):
            line = plt.Line2D([0], [0.5], color='black', linestyle=precision_linestyles[p_idx],
                              linewidth=1.2, marker=precision_markers[p_idx], markersize=precision_markersizes[p_idx],
                              markeredgecolor='black', markeredgewidth=0.8, markerfacecolor='none',
                              label=f'$\\epsilon$ = {precision}')
            precision_handles.append(line)
            precision_labels.append(f'$\\epsilon$ = {precision}')
        leg_prec = fig.legend(precision_handles, precision_labels, loc='center',
                              bbox_to_anchor=(0.50, -0.02), title='Precision',
                              frameon=True, fancybox=True, shadow=True)
        legends.append(leg_prec)

    # ----- Degree legend (row-wise ordering, ncol=3), identical to original -----
    sorted_legend_degrees = sorted(legend_degrees)
    n_deg = len(sorted_legend_degrees)
    degree_legend_ncol = 3
    degree_legend_nrow = int(np.ceil(n_deg / degree_legend_ncol))
    _deg_grid = [[None] * degree_legend_ncol for _ in range(degree_legend_nrow)]
    for idx, degree in enumerate(sorted_legend_degrees):
        r, c = idx // degree_legend_ncol, idx % degree_legend_ncol
        _deg_grid[r][c] = degree
    degree_legend_order = []
    for c in range(degree_legend_ncol):
        for r in range(degree_legend_nrow):
            if _deg_grid[r][c] is not None:
                degree_legend_order.append(_deg_grid[r][c])

    degree_legend_handles = []
    degree_legend_labels = []
    for degree in degree_legend_order:
        rect = mpatches.Rectangle((0, 0), 1, 1, facecolor=degree_color_map[degree],
                                  label=f'$d = {2 * degree}$')
        degree_legend_handles.append(rect)
        degree_legend_labels.append(f'$d = {2 * degree}$')

    # Shift the degree legend right in rotation-decomposition mode to make room
    # for the centred precision legend.
    degree_anchor_x = 0.80 if use_rotation_decomposition else 0.65
    leg2 = fig.legend(degree_legend_handles, degree_legend_labels, loc='center',
                      bbox_to_anchor=(degree_anchor_x, -0.02), title='Degree',
                      frameon=True, fancybox=True, shadow=True, ncol=degree_legend_ncol)
    legends.append(leg2)

    for leg in legends:
        fig.add_artist(leg)

    # In rotation-decomposition mode there are three legends (eigenvalue |
    # precision | degree). Reposition them as a group centred at x=0.5 with
    # equal horizontal gaps, using their measured widths.
    if use_rotation_decomposition:
        y_anchor = -0.02
        gap = 0.04  # figure-fraction gap between adjacent legends
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        widths = [leg.get_window_extent(renderer=renderer).width / fig.bbox.width
                  for leg in legends]
        total = sum(widths) + gap * (len(legends) - 1)
        cursor = 0.5 - total / 2
        for leg, w in zip(legends, widths):
            leg.set_bbox_to_anchor((cursor + w / 2, y_anchor), transform=fig.transFigure)
            cursor += w + gap


def main():
    # Usage: python QIPI/QIPI.py [--rot]
    #   --rot  plot the rotation-decomposition data (saves QIPI_with_rot_decompose.pdf).
    #          Default: normal QSVT data (saves QIPI.pdf).
    use_rotation_decomposition = '--rot' in sys.argv[1:]

    molecule_order = ['H2', 'LiH', 'BeH2']
    data = {m: load_data(m, use_rotation_decomposition) for m in molecule_order}

    fig, axes = plt.subplots(3, 2, figsize=(16, 18), dpi=100)

    # LaTeX chemical formulas for the row labels
    molecule_labels = {
        'H2': r'$\mathrm{H}_2$',
        'LiH': r'$\mathrm{LiH}$',
        'BeH2': r'$\mathrm{BeH}_2$',
    }

    for row, molecule in enumerate(molecule_order):
        ax_poly, ax_conv = axes[row, 0], axes[row, 1]
        plot_polynomial(ax_poly, data[molecule], use_rotation_decomposition)
        plot_convergence(ax_conv, data[molecule], use_rotation_decomposition)

    # Shared legend uses BeH2 as the reference for the target-eigenvalue color
    build_shared_legends(fig, data['BeH2'], use_rotation_decomposition)

    plt.tight_layout(pad=2.0)

    # Row labels, written vertically to the right of the rightmost (convergence)
    for row, molecule in enumerate(molecule_order):
        ax_conv = axes[row, 1]
        pos_conv = ax_conv.get_position()
        center_y = (pos_conv.y0 + pos_conv.y1) / 2
        fig.text(pos_conv.x1 + 0.012, center_y, molecule_labels[molecule],
                 ha='left', va='center', rotation=270,
                 fontsize=matplotlib.rcParams['axes.titlesize'])
    if use_rotation_decomposition:
        out_path = os.path.join(_REPO_ROOT, 'paper_plots', 'figure10.pdf')
    else:
        out_path = os.path.join(_REPO_ROOT, 'paper_plots', 'figure8.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
