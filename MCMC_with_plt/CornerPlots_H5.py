import os

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np


def corner_plot(galaxy_name):
    # Construct the path to the HDF5 backend
    filename = os.path.join("chains", f"chains_{galaxy_name}_rotmod.h5")

    # Load the backend
    backend = emcee.backends.HDFBackend(filename)

    # Extract the chain and log probabilities
    samples = backend.get_chain()

    print("Number of Iterations:", backend.iteration)
    print("Shape of chain:", samples.shape)

    # Determine dimensions dynamically
    ndim = samples.shape[2]

    # Updated parameter labels
    labels = ["r1", "log10(M200)", "c"]

    # Trace Plots: Setup subplots to match the number of dimensions
    fig, axes = plt.subplots(ndim, figsize=(8, 2 * ndim), sharex=True)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Stepps")
    plt.tight_layout()
    plt.show()

    # Apply a 30% burn-in
    burnin = int(0.3 * backend.iteration)

    # Flatten the chain after discarding burn-in
    flat_samples = backend.get_chain(discard=burnin, flat=True)

    # Corner Plot: includes r1, log10(M200), and c
    print(f"Generating corner plot for {ndim} parameters...")
    corner.corner(
        flat_samples,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt=".3f",
    )

    plt.show()


# Example usage for 'DDO170_rotmod':
# corner_plot('DDO170')
