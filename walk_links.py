import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# Set plotting style
sns.set_style("whitegrid")


def get_data(snaps, filepath, level):

    # Set up dictionaries to store direct progenitors and descendants
    reals = {}
    nparts = {}
    progs = {}
    descs = {}

    # Open files
    hdf = h5py.File(filepath, "r")

    # Loop over snapshots
    for snap in snaps:

        print("Extracting data for snap %s" % snap)

        # Open snapshot group
        if level == 0:
            snap_grp = hdf[snap]
        else:
            snap_grp = hdf[snap]["Subhalos"]

        # Define dict key
        key = int(snap)

        # Extract the realness flags
        reals[key] = snap_grp["real_flag"][...]

        # Extract nparts
        nparts[key] = snap_grp["nparts"][...]

        # Extract the start indices for each halo
        prog_start_index = snap_grp["prog_start_index"][...]
        desc_start_index = snap_grp["desc_start_index"][...]

        # Extract the direct links
        # (these are stored in the start_index position)
        progs[key] = np.full(nparts[key].size, -1)
        descs[key] = np.full(nparts[key].size, -1)
        okinds = np.logical_and(prog_start_index < 2 ** 30,
                                prog_start_index >= 0)
        progs[key][okinds] = snap_grp["Prog_haloIDs"][...][
            prog_start_index[okinds]]
        okinds = np.logical_and(desc_start_index < 2 ** 30,
                                desc_start_index >= 0)
        descs[key][okinds] = snap_grp["Desc_haloIDs"][...][
            desc_start_index[okinds]]

        print(snap, progs[key])

    hdf.close()

    return reals, nparts, progs, descs


def get_main_branch_lengths(reals, nparts, progs, descs):

    # How many halos?
    nhalo = nparts[98].size

    # Initialise array to store lengths
    lengths = np.zeros(nhalo, dtype=int)

    # Loop over halos at z0 (snap 0098)
    for ihalo in range(nhalo):

        # print("Walking main branch of halo %d (of %d)" % (ihalo, nhalo))

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


def get_persist_lengths(reals, nparts, progs, descs):

    # Set up search set
    done_halos = set()

    # Define the full length (initially nsnapshot + 1 long)
    full_length = 101

    # Initialise array to store lengths
    root_snaps = []
    lengths = []
    max_nparts = []
    max_snaps = []

    # Loop over snapshots
    for root_snap in range(0, 99):

        # Decrement the full length since we have moved forward in time
        full_length -= 1

        # Loop over halos in the current snapshot
        for ihalo in range(len(descs[root_snap])):

            # print("Walking main branch of halo %d in snap %s (of %d)"
            #       % (ihalo, str(root_snap).zfill(4), len(descs[root_snap])))

            # Skip if this halo is not real
            if not reals[root_snap][ihalo]:
                continue

            # Skip if this halo has already appeared in a main branch
            if (root_snap, ihalo) in done_halos:
                continue

            # Initialise looping variables
            desc = ihalo
            snap = root_snap
            length = 0
            max_npart = 0

            # Loop until there is no descendant (desc == -1)
            # or we reach the present day
            while desc != -1 and snap < 99:

                # Increment length
                length += 1

                # Get max npart
                if nparts[snap][desc] > max_npart:
                    max_npart = nparts[snap][desc]
                    max_snap = snap

                # Include this halo in done halos
                done_halos.update({(snap, desc), })

                # Get this halos direct progenitor
                desc = descs[snap][desc]

                # Decement snapshot
                snap += 1

            # Appended this halos persistence length
            root_snaps.append(root_snap)
            lengths.append(length)
            max_nparts.append(max_npart)
            max_snaps.append(max_snap)

    # Convert to arrays
    root_snaps = np.array(root_snaps)
    lengths = np.array(lengths)
    max_nparts = np.array(max_nparts)
    max_snaps = np.array(max_snaps)

    return root_snaps, lengths, max_nparts, max_snaps


def main_branch_length():

    # Open the snapshot list
    snaps = np.loadtxt("../graphs/L0100N0285_DMO/snaplist.txt", dtype=str)

    # Lets get the file paths
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    # And get the data from these files
    print("Reading DMO Hosts")
    reals_dmo, nparts_dmo, progs_dmo, descs_dmo = get_data(snaps, file1,
                                                           level=0)
    print("Reading DM Hosts")
    reals_dm, nparts_dm, progs_dm, descs_dm = get_data(snaps, file2, level=0)
    print("Reading DM+Baryon Hosts")
    (reals_dmbary, nparts_dmbary, progs_dmbary,
     descs_dmbary) = get_data(snaps, file3, level=0)
    print("Reading DMO Subhalos")
    (sub_reals_dmo, sub_nparts_dmo, sub_progs_dmo,
     sub_descs_dmo) = get_data(snaps, file1, level=1)
    print("Reading DM Subhalos")
    (sub_reals_dm, sub_nparts_dm, sub_progs_dm,
     sub_descs_dm) = get_data(snaps, file2, level=1)
    print("Reading DM+Baryon Subhalos")
    (sub_reals_dmbary, sub_nparts_dmbary, sub_progs_dmbary,
     sub_descs_dmbary) = get_data(snaps, file3, level=1)

    # Walk mian branches measuring lengths
    print("Walking DMO Hosts")
    l_dmo = get_main_branch_lengths(reals_dmo, nparts_dmo, progs_dmo,
                                    descs_dmo)
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
    prev_max = [0, 0, 0]
    for lab, c in zip(["DMO", "DM", "DM+Baryons"], ["r", "b", "g"]):

        # Define varibales for plotting
        if lab == "DMO":
            npart = nparts_dmo[98]
            real = reals_dmo[98]
            l = l_dmo
            print(l)
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

        for (i, ax), low, up in zip(enumerate([ax3, ax2, ax1]),
                                    low_threshs, up_threshs):

            okinds = np.logical_and(npart >= low,
                                    npart < up)
            okinds = np.logical_and(okinds, real)
            ls = l[okinds]

            H, _ = np.histogram(ls, bins=bin_edges)

            ax.plot(bin_edges[:-1] + 0.5, H, label=lab, color=c)

            H_max = np.max(H)
            if H_max > prev_max[i]:
                ax.set_ylim(1, H_max + (0.2 * H_max))
            prev_max[i] = H_max

    # Label axes
    ax3.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$N$')

    # Annotate the mass bins
    ax1.text(0.05, 0.8, r'$N_{p}>1000$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax1.transAxes,
             horizontalalignment='left')
    ax2.text(0.05, 0.8, r'$1000\geq N_{p}>100$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax2.transAxes,
             horizontalalignment='left')
    ax3.text(0.05, 0.8, r'$100\geq N_{p}>20$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax3.transAxes,
             horizontalalignment='left')

    # Remove x axis from upper subplots
    ax1.tick_params(axis='x', bottom=False, left=False)
    ax2.tick_params(axis='x', bottom=False, left=False)

    ax3.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.35),
               fancybox=True, ncol=3)

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
    prev_max = [0, 0, 0]
    for lab, c in zip(["DMO", "DM", "DM+Baryons"],
                      ["r", "b", "g"]):

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

        for (i, ax), low, up in zip(enumerate([ax3, ax2, ax1]),
                                    low_threshs, up_threshs):

            okinds = np.logical_and(npart >= low,
                                    npart < up)
            okinds = np.logical_and(okinds, real)
            ls = l[okinds]

            H, _ = np.histogram(ls, bins=bin_edges)

            ax.plot(bin_edges[:-1] + 0.5, H, label=lab, color=c)

            H_max = np.max(H)
            if H_max > prev_max[i]:
                ax.set_ylim(1, H_max + (0.2 * H_max))
            prev_max[i] = H_max

    # Label axes
    ax3.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$N$')

    # Annotate the mass bins
    ax1.text(0.05, 0.8, r'$N_{p}>1000$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax1.transAxes,
             horizontalalignment='left')
    ax2.text(0.05, 0.8, r'$1000\geq N_{p}>100$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax2.transAxes,
             horizontalalignment='left')
    ax3.text(0.05, 0.8, r'$100\geq N_{p}>20$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax3.transAxes,
             horizontalalignment='left')

    # Remove x axis from upper subplots
    ax1.tick_params(axis='x', bottom=False, left=False)
    ax2.tick_params(axis='x', bottom=False, left=False)

    ax3.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.35),
               fancybox=True, ncol=3)

    # Save figure with a transparent background
    fig.savefig('plots/sub_mainbranchlengthcomp.png', bbox_inches="tight")
    plt.close(fig)


def persist_length():

    # Open the snapshot list
    snaps = np.loadtxt("../graphs/L0100N0285_DMO/snaplist.txt", dtype=str)

    # Define the number of snapshots
    nsnaps = len(snaps)

    # Lets get the file paths
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    # And get the data from these files
    print("Reading DMO Hosts")
    reals_dmo, nparts_dmo, progs_dmo, descs_dmo = get_data(snaps, file1,
                                                           level=0)
    print("Reading DM Hosts")
    reals_dm, nparts_dm, progs_dm, descs_dm = get_data(snaps, file2, level=0)
    print("Reading DM+Baryon Hosts")
    (reals_dmbary, nparts_dmbary, progs_dmbary,
     descs_dmbary) = get_data(snaps, file3, level=0)
    print("Reading DMO Subhalos")
    (sub_reals_dmo, sub_nparts_dmo, sub_progs_dmo,
     sub_descs_dmo) = get_data(snaps, file1, level=1)
    print("Reading DM Subhalos")
    (sub_reals_dm, sub_nparts_dm, sub_progs_dm,
     sub_descs_dm) = get_data(snaps, file2, level=1)
    print("Reading DM+Baryon Subhalos")
    (sub_reals_dmbary, sub_nparts_dmbary, sub_progs_dmbary,
     sub_descs_dmbary) = get_data(snaps, file3, level=1)

    # Walk mian branches measuring lengths
    print("Walking DMO Hosts")
    root_dmo, l_dmo, n_dmo, max_s_dmo = get_persist_lengths(reals_dmo,
                                                            nparts_dmo,
                                                            progs_dmo,
                                                            descs_dmo)
    print("Walking DM Hosts")
    root_dm, l_dm, n_dm, max_s_dm = get_persist_lengths(reals_dm,
                                                        nparts_dm,
                                                        progs_dm,
                                                        descs_dm)
    print("Walking DM+Baryon Hosts")
    (root_dmbary, l_dmbary, n_dmbary,
     max_s_dmbary) = get_persist_lengths(reals_dmbary, nparts_dmbary,
                                         progs_dmbary, descs_dmbary)
    print("Walking DMO Subhalos")
    (root_dmo_sub, l_dmo_sub,
     n_dmo_sub, max_s_dmo_sub) = get_persist_lengths(sub_reals_dmo,
                                                     sub_nparts_dmo,
                                                     sub_progs_dmo,
                                                     sub_descs_dmo)
    print("Walking DM Subhalos")
    (root_dm_sub, l_dm_sub,
     n_dm_sub, max_s_dm_sub) = get_persist_lengths(sub_reals_dm, sub_nparts_dm,
                                                   sub_progs_dm, sub_descs_dm)
    print("Walking DM+Baryon Subhalos")
    (root_dmbary_sub, l_dmbary_sub,
     n_dmbary_sub, max_s_dmbary_sub) = get_persist_lengths(sub_reals_dmbary,
                                                           sub_nparts_dmbary,
                                                           sub_progs_dmbary,
                                                           sub_descs_dmbary)

    # Create lists of lower and upper mass thresholds for histograms
    low_threshs = [0, 100, 1000]
    up_threshs = [100, 1000, np.inf]

    # Define bins for the lengths
    bin_edges = np.linspace(0, 99, nsnaps)

    # Set up figure
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=3, ncols=1)
    gs.update(wspace=0.5, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Loop over simulations
    prev_max = [0, 0, 0]
    for lab, c in zip(["DMO", "DM", "DM+Baryons"], ["r", "b", "g"]):

        # Define varibales for plotting
        if lab == "DMO":
            npart = n_dmo
            l = (nsnaps - root_dmo) - l_dmo
        elif lab == "DM":
            npart = n_dm
            l = (nsnaps - root_dm) - l_dm
        elif lab == "DM+Baryons":
            npart = n_dmbary
            l = (nsnaps - root_dmbary) - l_dmbary
        else:
            print("Something is very wrong")
            break

        for (i, ax), low, up in zip(enumerate([ax3, ax2, ax1]),
                                    low_threshs, up_threshs):

            ax.semilogy()

            okinds = np.logical_and(npart >= low,
                                    npart < up)
            ls = l[okinds]

            H, _ = np.histogram(ls, bins=bin_edges)

            ax.plot(bin_edges[:-1] + 0.5, H, label=lab, color=c)

            H_max = np.max(H)
            if H_max > prev_max[i]:
                ax.set_ylim(1, H_max + (0.2 * H_max))
            prev_max[i] = H_max

    # Label axes
    ax3.set_xlabel(r'$\ell_{p}$')
    ax2.set_ylabel(r'$N$')

    # Annotate the mass bins
    ax1.text(0.05, 0.8, r'$N_{p}>1000$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax1.transAxes,
             horizontalalignment='left')
    ax2.text(0.05, 0.8, r'$1000\geq N_{p}>100$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax2.transAxes,
             horizontalalignment='left')
    ax3.text(0.05, 0.8, r'$100\geq N_{p}>20$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax3.transAxes,
             horizontalalignment='left')

    # Remove x axis from upper subplots
    ax1.tick_params(axis='x', bottom=False, left=False)
    ax2.tick_params(axis='x', bottom=False, left=False)

    ax3.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.35),
               fancybox=True, ncol=3)

    # Save figure with a transparent background
    fig.savefig('plots/persistlengthcomp.png', bbox_inches="tight")
    plt.close(fig)

    # Set up figure
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=3, ncols=1)
    gs.update(wspace=0.5, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Loop over simulations
    prev_max = [0, 0, 0]
    for lab, c in zip(["DMO", "DM", "DM+Baryons"],
                      ["r", "b", "g"]):

        # Define varibales for plotting
        if lab == "DMO":
            npart = n_dmo_sub
            l = (nsnaps - root_dmo_sub) - l_dmo_sub
        elif lab == "DM":
            npart = n_dm_sub
            l = (nsnaps - root_dm_sub) - l_dm_sub
        elif lab == "DM+Baryons":
            npart = n_dmbary_sub
            l = (nsnaps - root_dmbary_sub) - l_dmbary_sub
        else:
            print("Something is very wrong")
            break

        for (i, ax), low, up in zip(enumerate([ax3, ax2, ax1]),
                                    low_threshs, up_threshs):

            ax.semilogy()

            okinds = np.logical_and(npart >= low,
                                    npart < up)
            ls = l[okinds]

            H, _ = np.histogram(ls, bins=bin_edges)

            ax.plot(bin_edges[:-1] + 0.5, H, label=lab, color=c)

            H_max = np.max(H)
            if H_max > prev_max[i]:
                ax.set_ylim(1, H_max + (0.2 * H_max))
            prev_max[i] = H_max

    # Label axes
    ax3.set_xlabel(r'$\ell$')
    ax2.set_ylabel(r'$N$')

    # Annotate the mass bins
    ax1.text(0.05, 0.8, r'$N_{p}>1000$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax1.transAxes,
             horizontalalignment='left')
    ax2.text(0.05, 0.8, r'$1000\geq N_{p}>100$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax2.transAxes,
             horizontalalignment='left')
    ax3.text(0.05, 0.8, r'$100\geq N_{p}>20$',
             bbox=dict(boxstyle="round,pad=0.3", fc='w',
                       ec="k", lw=1, alpha=0.8),
             transform=ax3.transAxes,
             horizontalalignment='left')

    # Remove x axis from upper subplots
    ax1.tick_params(axis='x', bottom=False, left=False)
    ax2.tick_params(axis='x', bottom=False, left=False)

    ax3.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.35),
               fancybox=True, ncol=3)

    # Save figure with a transparent background
    fig.savefig('plots/sub_persistlengthcomp.png', bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main_branch_length()
    persist_length()
