# Plot functions

from matplotlib import pyplot as plt

import numerics as bd


def plot_support_2d(l: list, rng: tuple[range]):
    """Plots where the given objects have non-zero elements (up to machine threshold) in rng.
    
    Args:
        l (list): A list of objects that support subscript operation.
        rng (tuple[range]): A 2D tuple of ranges defining the rectangle in the Z^2 grid to plot."""

    if len(rng) != 2:
        raise ValueError("rng must have dimension 2.")

    for k in range(len(l)):
        px, py = zip(*[
            (x+k*0.05, y+k*0.05) for x in rng[0] for y in rng[1] if bd.abs(l[k][x, y]) > bd.machine_threshold()
        ])
        
        plt.scatter(px, py, marker='o', label=f"Support #{k+1}")

    # Formatting
    plt.xlabel("k")
    plt.ylabel("h")
    plt.title("Supports")
    plt.grid(False)
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.show()