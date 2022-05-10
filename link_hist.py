import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# Set plotting style
sns.set_style("whitegrid")


def plot_link():

    # Open the snapshot list
    snaps = np.loadtxt("../graphs/L0100N0285_DMO/snaplist.txt", dtype=str)

    # Lets get the file paths
    ini_file1 = sys.argv[1]
    ini_file2 = sys.argv[2]
    ini_file3 = sys.argv[3]
    ini_sub_file1 = sys.argv[4]
    ini_sub_file2 = sys.argv[5]
    ini_sub_file3 = sys.argv[6]

    # Initialise containers for the data
    nprogs = {"DMO": {}, "DM": {}, "DM+Baryons": {}}
    ndescs = {"DMO": {}, "DM": {}, "DM+Baryons": {}}
    sub_nprogs = {"DMO": {}, "DM": {}, "DM+Baryons": {}}
    sub_ndescs = {"DMO": {}, "DM": {}, "DM+Baryons": {}}

    # Loop over snapshots
    for snap in snaps:

        try:

            # Define this snapshots file string
            file1 = ini_file1.replace("0098", snap)
            file2 = ini_file2.replace("0098", snap)
            file3 = ini_file3.replace("0098", snap)
            sub_file1 = ini_sub_file1.replace("0098", snap)
            sub_file2 = ini_sub_file2.replace("0098", snap)
            sub_file3 = ini_sub_file3.replace("0098", snap)

            # Open files
            hdf1 = h5py.File(file1, "r")
            hdf2 = h5py.File(file2, "r")
            hdf3 = h5py.File(file3, "r")

            # Open total progs
            nprogs["DMO"][snap] = hdf1["nProgs"][:]
            nprogs["DM"][snap] = hdf2["nProgs"][:]
            nprogs["DM+Baryons"][snap] = hdf3["nProgs"][:]
            ndescs["DMO"][snap] = hdf1["nDescs"][:]
            ndescs["DM"][snap] = hdf2["nDescs"][:]
            ndescs["DM+Baryons"][snap] = hdf3["nDescs"][:]

            hdf1.close()
            hdf2.close()
            hdf3.close()

            # Open files
            hdf1 = h5py.File(sub_file1, "r")
            hdf2 = h5py.File(sub_file2, "r")
            hdf3 = h5py.File(sub_file3, "r")

            # Open total progs
            sub_nprogs["DMO"][snap] = hdf1["nProgs"][:]
            sub_nprogs["DM"][snap] = hdf2["nProgs"][:]
            sub_nprogs["DM+Baryons"][snap] = hdf3["nProgs"][:]
            sub_ndescs["DMO"][snap] = hdf1["nDescs"][:]
            sub_ndescs["DM"][snap] = hdf2["nDescs"][:]
            sub_ndescs["DM+Baryons"][snap] = hdf3["nDescs"][:]

            hdf1.close()
            hdf2.close()
            hdf3.close()

        except (OSError, KeyError):
            print("Failed to open snap %s" % snap)
            continue

    # Combine into total arrays over all time
    progs_dmo = np.concatenate(list(nprogs["DMO"].values()))
    progs_dm = np.concatenate(list(nprogs["DM"].values()))
    progs_dmbary = np.concatenate(list(nprogs["DM+Baryons"].values()))
    descs_dmo = np.concatenate(list(ndescs["DMO"].values()))
    descs_dm = np.concatenate(list(ndescs["DM"].values()))
    descs_dmbary = np.concatenate(list(ndescs["DM+Baryons"].values()))

    sub_progs_dmo = np.concatenate(list(sub_nprogs["DMO"].values()))
    sub_progs_dm = np.concatenate(list(sub_nprogs["DM"].values()))
    sub_progs_dmbary = np.concatenate(list(sub_nprogs["DM+Baryons"].values()))
    sub_descs_dmo = np.concatenate(list(sub_ndescs["DMO"].values()))
    sub_descs_dm = np.concatenate(list(sub_ndescs["DM"].values()))
    sub_descs_dmbary = np.concatenate(list(sub_ndescs["DM+Baryons"].values()))

    # Get maximum number of links
    link_max = np.max((progs_dmo, progs_dm, progs_dmbary,
                       descs_dmo, descs_dm, descs_dmbary,
                       sub_progs_dmo, sub_progs_dm, sub_progs_dmbary,
                       sub_descs_dmo, sub_descs_dm, sub_descs_dmbary))

    # Define bins
    bins = np.arange(0, link_max + 1, 1)
    bin_cents = (bins[1:] + bins[:-1]) / 2

    # Histogram progs
    prog_H_dmo, _ = np.histogram(progs_dmo, bins=bins)
    prog_H_dm, _ = np.histogram(progs_dm, bins=bins)
    prog_H_dmbary, _ = np.histogram(progs_dmbary, bins=bins)
    desc_H_dmo, _ = np.histogram(descs_dmo, bins=bins)
    desc_H_dm, _ = np.histogram(descs_dm, bins=bins)
    desc_H_dmbary, _ = np.histogram(descs_dmbary, bins=bins)
    sub_prog_H_dmo, _ = np.histogram(sub_progs_dmo, bins=bins)
    sub_prog_H_dm, _ = np.histogram(sub_progs_dm, bins=bins)
    sub_prog_H_dmbary, _ = np.histogram(sub_progs_dmbary, bins=bins)
    sub_desc_H_dmo, _ = np.histogram(sub_descs_dmo, bins=bins)
    sub_desc_H_dm, _ = np.histogram(sub_descs_dm, bins=bins)
    sub_desc_H_dmbary, _ = np.histogram(sub_descs_dmbary, bins=bins)

    # Set up plot
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1)
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax.semilogy()
    ax1.semilogy()

    # Plot curves
    ax.plot(bin_cents, prog_H_dmo, label="DMO", color="r")
    ax.plot(bin_cents, prog_H_dm, label="DM", color="b")
    ax.plot(bin_cents, prog_H_dmbary,
            label="DM+Baryons", color="g")
    ax.plot(bin_cents, desc_H_dmo, color="r", linestyle="--")
    ax.plot(bin_cents, desc_H_dm, color="b", linestyle="--")
    ax.plot(bin_cents, desc_H_dmbary,
            color="g", linestyle="--")

    ax1.plot(bin_cents, sub_prog_H_dmo, label="DMO", color="r")
    ax1.plot(bin_cents, sub_prog_H_dm, label="DM", color="b")
    ax1.plot(bin_cents, sub_prog_H_dmbary,
             label="DM+Baryons", color="g")
    ax1.plot(bin_cents, sub_desc_H_dmo, color="r", linestyle="--")
    ax1.plot(bin_cents, sub_desc_H_dm, color="b", linestyle="--")
    ax1.plot(bin_cents, sub_desc_H_dmbary,
             color="g", linestyle="--")

    ax.plot([1000, 1001], [0, 1], color="k", linestyle="-",
            label="Progenitor")
    ax.plot([1000, 1001], [0, 1], color="k", linestyle="--",
            label="Descendant")

    # Label axes
    ax1.set_xlabel(r"$N_{\mathrm{link}}$")
    ax.set_ylabel(r"$N$")
    ax1.set_ylabel(r"$N$")

    # Set limits
    ax.set_xlim(0, link_max + 4)
    ax1.set_xlim(0, link_max + 4)
    ax.set_ylim(0, None)

    # Draw legend
    ax.legend()

    fig.savefig("plots/prog_desc_hist_%s.png" % snap,
                bbox_inches="tight")
    plt.close(fig)

    # # Calculate percentiles for evolution
    # zs = []
    # medians_dmo = []
    # pcent_16_dmo = []
    # pcent_84_dmo = []
    # medians_dm = []
    # pcent_16_dm = []
    # pcent_84_dm = []
    # medians_dmbary = []
    # pcent_16_dmbary = []
    # pcent_84_dmbary = []
    # for snap in snaplist:

    #     # Calculate percentiles
    #     medians_dmo.append()


if __name__ == "__main__":
    plot_link()
