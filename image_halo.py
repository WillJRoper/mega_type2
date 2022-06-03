import h5py
import numpy as np
import matplotlib.pyplot as plt


def make_img():

    # Define constants
    path = "../graphs/L0100N0285_DMO/data/halos/DMO_halos_0092.hdf5"
    simpath = "../runs/L0100N0285/data/L0100N0285_DMO_"
    halo = 67278
    snaps = ["0092", "0093", "0094"]
    width = 0.2  # cMpc
    soft = 0.01754385964

    # Get this halos data from the halo file
    hdf = h5py.File(path, "r")
    boxsize = hdf.attrs["boxsize"]
    begin = hdf["start_index"][halo]
    length = hdf["stride"][halo]
    cent = hdf["mean_positions"][halo, :]
    print(halo, begin, length, cent)
    sim_pids = hdf["sim_part_ids"][begin: begin + length]
    hdf.close()

    # Set up plot
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Loop over snaps
    for ax, snap in zip([ax1, ax2, ax3], snaps):

        # Get sim data
        hdf = h5py.File(simpath + snap + ".hdf5", "r")
        pos = hdf["PartType1"]["Coordinates"][...]
        pids = hdf["PartType1"]["ParticleIDs"][...]
        hdf.close()

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

    fig.savefig("plots/anomalous_halo_halo%d_snap%s.png" % (halo, snaps[-1]))


if __name__ == "__main__":
    make_img()
