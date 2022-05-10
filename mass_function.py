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

    # Extract the realness flags
    reals1 = hdf1["real_flag"][...]
    reals2 = hdf2["real_flag"][...]
    reals3 = hdf3["real_flag"][...]
    sub_reals1 = hdf1["Subhalos"]["real_flag"][...]
    sub_reals2 = hdf2["Subhalos"]["real_flag"][...]
    sub_reals3 = hdf3["Subhalos"]["real_flag"][...]

    # Open total masses
    if part_type is None:
        masses_dmo = hdf1["masses"][reals1] * 10 ** 10
        masses_dm = hdf2["masses"][reals2] * 10 ** 10
        masses_dmbary = hdf3["masses"][reals3] * 10 ** 10
        sub_masses_dmo = hdf1["Subhalos"]["masses"][sub_reals1] * 10 ** 10
        sub_masses_dm = hdf2["Subhalos"]["masses"][sub_reals2] * 10 ** 10
        sub_masses_dmbary = hdf3["Subhalos"]["masses"][sub_reals3] * 10 ** 10
    else:
        masses_dmo = hdf1["part_type_masses"][:, part_type][reals1] * 10 ** 10
        masses_dm = hdf2["part_type_masses"][:, part_type][reals2] * 10 ** 10
        masses_dmbary = hdf3["part_type_masses"][:,
                                                 part_type][reals3] * 10 ** 10
        sub_masses_dmo = hdf1["Subhalos"][
            "part_type_masses"][:, part_type][reals1] * 10 ** 10
        sub_masses_dm = hdf2["Subhalos"][
            "part_type_masses"][:, part_type][reals2] * 10 ** 10
        sub_masses_dmbary = hdf3["Subhalos"][
            "part_type_masses"][:, part_type][reals3] * 10 ** 10

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
    sub_H_dmo, _ = np.histogram(sub_masses_dmo, bins=bins)
    sub_H_dm, _ = np.histogram(sub_masses_dm, bins=bins)
    sub_H_dmbary, _ = np.histogram(sub_masses_dmbary, bins=bins)

    # Convert units to per unit mass per volume
    phi_dmo = H_dmo / intervals / vol
    phi_dm = H_dm / intervals / vol
    phi_dmbary = H_dmbary / intervals / vol
    sub_phi_dmo = sub_H_dmo / intervals / vol
    sub_phi_dm = sub_H_dm / intervals / vol
    sub_phi_dmbary = sub_H_dmbary / intervals / vol

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.loglog()

    # Plot curves
    okinds = phi_dmo > 0
    ax.plot(bin_cents[okinds], phi_dmo[okinds], label="DMO", color="r")
    okinds = phi_dm > 0
    ax.plot(bin_cents[okinds], phi_dm[okinds], label="DM", color="b")
    okinds = phi_dmbary > 0
    ax.plot(bin_cents[okinds], phi_dmbary[okinds],
            label="DM+Baryons", color="g")
    okinds = sub_phi_dmo > 0
    ax.plot(bin_cents[okinds], sub_phi_dmo[okinds], color="r", linestyle="--")
    okinds = sub_phi_dm > 0
    ax.plot(bin_cents[okinds], sub_phi_dm[okinds], color="b", linestyle="--")
    okinds = sub_phi_dmbary > 0
    ax.plot(bin_cents[okinds], sub_phi_dmbary[okinds],
            color="g", linestyle="--")
    ax.plot([0, 1], [0, 1], color="k", linestyle="-", label="Host")
    ax.plot([0, 1], [0, 1], color="k", linestyle="--", label="Subhalo")

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

    # Set limits
    ax.set_xlim(10**9, 10**15)
    ax.set_ylim(10**-19.5, 10**-10.5)

    # Draw legend
    ax.legend()

    if part_type is None:
        fig.savefig("plots/mass_function_total_%s.png" % snap,
                    bbox_inches="tight")
    else:
        fig.savefig("plots/mass_function_PartType%d_%s.png" % (part_type,
                                                               snap),
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_mass_funcs("0098", part_type=None)
    plot_mass_funcs("0098", part_type=1)
    plot_mass_funcs("0098", part_type=0)
