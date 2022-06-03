import h5py
import numpy as np
import matplotlib.pyplot as plt


def make_img():

    # Define constants
    path = "../graphs/L0100N0285_DMO/data/halos/DMO_halos_0092.hdf5"
    simpath = "../runs/L0100N0285/data/L0100N0285_DMO_"
    halo = 67278
    snaps = ["0092", "0093", "0094"]
    width = 1  # cMpc
    soft = 0.01754385964

    # Get this halos data from the halo file
    hdf = h5py.File(path, "r")
    boxsize = hdf.attrs["boxsize"][0]
    begin = hdf["start_index"][halo]
    length = hdf["stride"][halo]
    print(halo, begin, length)
    sim_pids = hdf["sim_part_ids"][begin: begin + length]
    hdf.close()

    # Set up plot
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Loop over snaps
    cent = np.array([0, 0, 0])
    for ax, snap in zip([ax1, ax2, ax3], snaps):

        # Get sim data
        hdf = h5py.File(simpath + snap + ".hdf5", "r")
        pos = hdf["PartType1"]["Coordinates"][...]
        pids = hdf["PartType1"]["ParticleIDs"][...]
        hdf.close()

        # Get indices of halo particles
        halo_inds = np.in1d(pids, sim_pids)

        # Define the center
        if snap == snaps[0]:
            cent = np.mean(pos[halo_inds, :], axis=0)

        # Shift and wrap the positions
        pos -= cent
        pos[pos < -boxsize / 2] += boxsize
        pos[pos > boxsize / 2] -= boxsize

        # Plot the background field
        H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1],
                                 bins=int(width / soft),
                                 range=((-width / 2, width / 2),
                                        (-width / 2, width / 2)))

        ax.imshow(H, extent=(-width / 2, width / 2, -width / 2, width / 2))

        # Get this halos data from this snapshot
        hdf = h5py.File(path.replace("0092", snap), "r")
        halo_ids = hdf["particle_halo_IDs"][...]
        all_sim_pids = hdf["all_sim_part_ids"][...]
        hdf.close()

        # Get indices of halo particles
        halo_inds = np.in1d(all_sim_pids, sim_pids)
        print(sim_pids)
        part_halo_ids = halo_ids[halo_inds]

        print(snap, np.unique(part_halo_ids, return_counts=True))

    fig.savefig("plots/anomalous_halo_halo%d_snap%s.png" % (halo, snaps[0]))


if __name__ == "__main__":
    make_img()
