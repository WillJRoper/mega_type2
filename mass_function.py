import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set plotting style
sns.set_style("whitegrid")


def plot_mass_funcs(snap, part_type=None):

    # Lets get the file paths
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    # Open files
    hdf1 = h5py.File(file1, "r")
    hdf2 = h5py.File(file2, "r")
    hdf3 = h5py.File(file3, "r")

    # Get box volume
    l = hdf1.attrs["boxsize"]
    vol = l[0] * l[1] * l[2]

    # Open total masses
    if part_type is None:
        masses_dmo = hdf1["masses"][:] * 10 ** 10
        masses_dm = hdf2["masses"][:] * 10 ** 10
        masses_dmbary = hdf3["masses"][:] * 10 ** 10
    else:
        masses_dmo = hdf1["part_type_masses"][:, part_type] * 10 ** 10
        masses_dm = hdf2["part_type_masses"][:, part_type] * 10 ** 10
        masses_dmbary = hdf3["part_type_masses"][:, part_type] * 10 ** 10

    hdf1.close()
    hdf2.close()
    hdf3.close()

    # Define bins
    bins = np.logspace(8, 16, 50)
    bin_cents = (bins[1:] + bins[:-1]) / 2
    intervals = bins[1:] - bins[:-1]

    # Histogram masses
    H_dmo, _ = np.histogram(masses_dmo, bins=bins)
    H_dm, _ = np.histogram(masses_dm, bins=bins)
    H_dmbary, _ = np.histogram(masses_dmbary, bins=bins)

    # Convert units to per unit mass per volume
    phi_dmo = H_dmo / intervals / vol
    phi_dm = H_dm / intervals / vol
    phi_dmbary = H_dmbary / intervals / vol

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.loglog()

    # Plot curves
    ax.plot(bin_cents, phi_dmo, label="DMO")
    ax.plot(bin_cents, phi_dm, label="DM")
    ax.plot(bin_cents, phi_dmbary, label="DM+Baryons")

    # Label axes
    if part_type is None:
        ax.set_xlabel(r"$M_{\mathrm{tot}}/M_\odot$")
    elif part_type == 1:
        ax.set_xlabel(r"$M_{\mathrm{DM}}/M_\odot$")
    elif part_type == 0:
        ax.set_xlabel(r"$M_{\mathrm{gas}}/M_\odot$")
    else:
        print("No such part_type=%d" % part_type)
    ax.set_ylabel(r"$\phi / [\mathrm{Mpc}^{-3} M_\odot]$")

    # Draw legend
    ax.legend()

    if part_type is None:
        fig.savefig("plots/mass_function_total_%s.png" % snap,
                    bbox_inches="tight")
    else:
        fig.savefig("plots/mass_function_PartType%d_%s.png" % (part_type, snap),
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_mass_funcs("0098", part_type=None)
    plot_mass_funcs("0098", part_type=1)
    plot_mass_funcs("0098", part_type=0)
