import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# Set plotting style
sns.set_style("whitegrid")


def get_data(snaps, ini_filepath):

    # Set up dictionaries to store direct progenitors and descendants
    reals = {}
    nparts = {}
    progs = {}
    descs = {}

    # Loop over snapshots
    for snap in snaps:

        print("Extracting data for snap %s" % snap)

        # Define this snapshots file string
        filepath = ini_filepath.replace("0098", snap)

        # Define dict key
        key = int(snap)

        # Open files
        hdf = h5py.File(filepath, "r")

        # Extract the realness flags
        reals[key] = hdf["real_flag"][...]

        # Extract nparts
        nparts[key] = hdf["nparts"][...]

        # Extract the start indices for each halo
        prog_start_index = hdf["prog_start_index"][...]
        desc_start_index = hdf["desc_start_index"][...]

        # Extract the direct links
        # (these are stored in the start_index position)
        progs[key] = np.full(nparts[key].size, -1)
        descs[key] = np.full(nparts[key].size, -1)
        okinds = prog_start_index < 2 ** 30
        progs[key][okinds] = hdf["Prog_haloIDs"][...][prog_start_index[okinds]]
        okinds = desc_start_index < 2 ** 30
        descs[key][okinds] = hdf["Desc_haloIDs"][...][desc_start_index[okinds]]

        hdf.close()

    return reals, nparts, progs, descs


def get_main_branch_lengths(reals, nparts, progs, descs):

    # How many halos?
    nhalo = nparts["0098"].size

    # Initialise array to store lengths
    lengths = np.zeros(nhalo, dtype=int)

    # Loop over halos at z0 (snap 0098)
    for ihalo in range(nhalo):

        print("Walking main branch of halo %d (of %d)" % (ihalo, nhalo))

        # Initialise looping variables
        prog = ihalo
        snap = 98

        # Loop until there is no progenitor (prog == -1)
        while prog != -1 and snap >= 0:

            # Increment length
            lengths[ihalo] += 1

            # Get this halos direct progenitor
            prog = progs[snap][prog]

            # Decement snapshot
            snap -= 1

    return lengths


def main_branch_length():

    # Open the snapshot list
    snaps = np.loadtxt("../graphs/L0100N0285_DMO/snaplist.txt", dtype=str)

    # Lets get the file paths
    ini_file1 = sys.argv[1]
    ini_file2 = sys.argv[2]
    ini_file3 = sys.argv[3]
    sub_ini_file1 = sys.argv[4]
    sub_ini_file2 = sys.argv[5]
    sub_ini_file3 = sys.argv[6]

    # And get the data from these files
    print("Reading DMO Hosts")
    reals_dmo, nparts_dmo, progs_dmo, descs_dmo = get_data(snaps, ini_file1)
    print("Reading DM Hosts")
    reals_dm, nparts_dm, progs_dm, descs_dm = get_data(snaps, ini_file2)
    print("Reading DM+Baryon Hosts")
    (reals_dmbary, nparts_dmbary, progs_dmbary,
     descs_dmbary) = get_data(snaps, ini_file3)
    print("Reading DMO Subhalos")
    (sub_reals_dmo, sub_nparts_dmo, sub_progs_dmo,
     sub_descs_dmo) = get_data(snaps, sub_ini_file1)
    print("Reading DM Subhalos")
    (sub_reals_dm, sub_nparts_dm, sub_progs_dm,
     sub_descs_dm) = get_data(snaps, sub_ini_file2)
    print("Reading DM+Baryon Subhalos")
    (sub_reals_dmbary, sub_nparts_dmbary, sub_progs_dmbary,
     sub_descs_dmbary) = get_data(snaps, sub_ini_file3)

    # Walk mian branches measuring lengths
    print("Walking DMO Hosts")
    l_dmo = get_main_branch_lengths(
        reals_dmo, nparts_dmo, progs_dmo, descs_dmo)
    print("Walking DM Hosts")
    l_dm = get_main_branch_lengths(reals_dm, nparts_dm, progs_dm, descs_dm)
    print("Walking DM+Baryon Hosts")
    l_dmbary = get_main_branch_lengths(reals_dmbary, nparts_dmbary,
                                       progs_dmbary, descs_dmbary)
    print("Walking DMO Subhalos")
    l_dmo_sub = get_main_branch_lengths(sub_reals_dmo, sub_nparts_dmo,
                                        sub_progs_dmo, sub_descs_dmo)
    print("Walking DM Subhalos")
    l_dm_sub = get_main_branch_lengths(sub_reals_dm, sub_nparts_dm,
                                       sub_progs_dm, sub_descs_dm)
    print("Walking DM+Baryon Subhalos")
    l_dmbary_sub = get_main_branch_lengths(sub_reals_dmbary, sub_nparts_dmbary,
                                           sub_progs_dmbary, sub_descs_dmbary)

    # Create lists of lower and upper mass thresholds for histograms
    low_threshs = [0, 100, 1000]
    up_threshs = [100, 1000, np.inf]

    # Define bins for the lengths
    bin_edges = np.linspace(0, 99, 100)

    # Set up figure
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=3, ncols=1)
    gs.update(wspace=0.5, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Loop over simulations
    for lab in ["DMO", "DM", "DM+Baryons"]:

        # Define varibales for plotting
        if lab == "DMO":
            npart = nparts_dmo[98]
            real = reals_dmo[98]
            l = l_dmo
        elif lab == "DM":
            npart = nparts_dm[98]
            real = reals_dm[98]
            l = l_dm
        elif lab == "DM+Baryons":
            npart = nparts_dmbary[98]
            real = reals_dmbary[98]
            l = l_dmbary
        else:
            print("Something is very wrong")
            break

        for ax, low, up in zip([ax3, ax2, ax1], low_threshs, up_threshs):

            okinds = np.logical_and(npart >= low,
                                    npart < up)
            okinds = np.logical_and(okinds, real)
            ls = l[okinds]

            H, _ = np.histogram(ls, bins=bin_edges)

            ax.plot(bin_edges[:-1] + 0.5, H, label=lab)

    # Label axes
    ax3.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$N$')

    # Annotate the mass bins
    ax1.text(0.05, 0.8, r'$M_{H}>1000$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax1.transAxes,
             horizontalalignment='left')
    ax2.text(0.05, 0.8, r'$1000\geq M_{H}>100$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax2.transAxes,
             horizontalalignment='left')
    ax3.text(0.05, 0.8, r'$100\geq M_{H}>20$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax3.transAxes,
             horizontalalignment='left')

    # Remove x axis from upper subplots
    ax1.tick_params(axis='x', bottom=False, left=False)
    ax2.tick_params(axis='x', bottom=False, left=False)

    # Set y axis limits such that 0 is removed from the upper two subplots to
    # avoid tick stacking
    ax1.set_ylim(0.5, None)
    ax2.set_ylim(0.5, None)

    # Save figure with a transparent background
    fig.savefig('plots/mainbranchlengthcomp.png', bbox_inches="tight")
    plt.close(fig)

    # Set up figure
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=3, ncols=1)
    gs.update(wspace=0.5, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Loop over simulations
    for lab in ["DMO", "DM", "DM+Baryons"]:

        # Define varibales for plotting
        if lab == "DMO":
            npart = sub_nparts_dmo[98]
            real = sub_reals_dmo[98]
            l = l_dmo_sub
        elif lab == "DM":
            npart = sub_nparts_dm[98]
            real = sub_reals_dm[98]
            l = l_dm_sub
        elif lab == "DM+Baryons":
            npart = sub_nparts_dmbary[98]
            real = sub_reals_dmbary[98]
            l = l_dmbary_sub
        else:
            print("Something is very wrong")
            break

        for ax, low, up in zip([ax3, ax2, ax1], low_threshs, up_threshs):

            okinds = np.logical_and(npart >= low,
                                    npart < up)
            okinds = np.logical_and(okinds, real)
            ls = l[okinds]

            H, _ = np.histogram(ls, bins=bin_edges)

            ax.plot(bin_edges[:-1] + 0.5, H, label=lab)

    # Label axes
    ax3.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$N$')

    # Annotate the mass bins
    ax1.text(0.05, 0.8, r'$M_{H}>1000$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax1.transAxes,
             horizontalalignment='left')
    ax2.text(0.05, 0.8, r'$1000\geq M_{H}>100$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax2.transAxes,
             horizontalalignment='left')
    ax3.text(0.05, 0.8, r'$100\geq M_{H}>20$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax3.transAxes,
             horizontalalignment='left')

    # Remove x axis from upper subplots
    ax1.tick_params(axis='x', bottom=False, left=False)
    ax2.tick_params(axis='x', bottom=False, left=False)

    # Set y axis limits such that 0 is removed from the upper two subplots to
    # avoid tick stacking
    ax1.set_ylim(0.5, None)
    ax2.set_ylim(0.5, None)

    # Save figure with a transparent background
    fig.savefig('plots/sub_mainbranchlengthcomp.png', bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main_branch_length()