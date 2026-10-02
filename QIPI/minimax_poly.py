import numpy as np
from scipy.special import chebyt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

'Eigenstate filtering polynomials with different degrees d for ∆ = 0.15'

plt.style.use('./plotstylefile.mplstyle')

Del = 0.15

def optpoly_eigenstatefiltering(x, l, Delta):

    'Eigenstate filtering polynomial with degree l for ∆ = 0.15'

    arg_top = -1 + 2 * ((x**2 - Delta**2) / (1 - Delta**2))
    arg_bot = -1 + 2 * ((-Delta**2) / (1 - Delta**2))
    cheb_l = chebyt(l)
    return cheb_l(arg_top) / cheb_l(arg_bot)



fig, ax1 = plt.subplots(figsize=(9.5, 7.7), dpi=100)
ax1.axvspan(-Del, Del, color='grey', alpha=0.4)
ax1.text(
    0,
    1.02,
    r'$\leftarrow\Delta = ' + str(np.round(Del, 2)) + r'\rightarrow$',
    ha='center',
    va='bottom',
    transform=ax1.get_xaxis_transform(),
)
colorit = 0
x_range = np.arange(-1, 1.001, 0.001)
degree_handles = []

for i in 8, 16, 32:
    color = "C" + str(colorit)
    colorit += 1
    (deg_line,) = ax1.plot(
        x_range,
        optpoly_eigenstatefiltering(x_range, i, Del),
        color=color,
        label=r'$d=$' + str(2 * i),
    )
    degree_handles.append(deg_line)

    ax1.axhline(y=2 * np.exp(-np.sqrt(2) * i * Del), color=color, linestyle="dashdot")
    ax1.axhline(y=-2 * np.exp(-np.sqrt(2) * i * Del), color=color, linestyle="dashdot")
    ax1.axhline(y=optpoly_eigenstatefiltering(Del, i, Del), color=color, linestyle="dotted")
    ax1.axhline(y=-optpoly_eigenstatefiltering(Del, i, Del), color=color, linestyle="dotted")

line = Line2D([0], [0], label='Theo. bound', linestyle="dashdot", color='k')
line2 = Line2D([0], [0], label='Num. bound', linestyle="dotted", color='k')


d0, d1, d2 = degree_handles
handles = [d0, line, d1, line2, d2]

ax1.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.18),
    ncols=3,
    frameon=True,
    fancybox=True,
    shadow=True,
)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$R_d(x,\Delta)$')
ax1.set_ylim(-0.25, 1.15)

plt.tight_layout(rect=[0, 0.06, 1, 1])

plotdir = "./paper_plots/"
plt.savefig(
    plotdir + "figure4.pdf",
    dpi=300,
    bbox_inches='tight',
)
# plt.show()
