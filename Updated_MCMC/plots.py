import os
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

def corner_plot_sidm(full_filename):
    """
    Aufruf: corner_plot_sidm('sidm_Test_DDO154_nw_16.h5')
    """
    # Pfad zum Ordner 'chains' hinzufügen
    filename = os.path.join("chains", full_filename)

    if not os.path.exists(filename):
        print(f"Datei nicht gefunden: {filename}")
        return

    # Load the backend
    backend = emcee.backends.HDFBackend(filename)

    # ... Rest der Funktion bleibt gleich ...
    samples = backend.get_chain()
    ndim = samples.shape[2]
    labels = ["r1", "log10(M200)", "c"]

    # Trace Plots
    fig, axes = plt.subplots(ndim, figsize=(8, 2 * ndim), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Steps")
    plt.tight_layout()
    plt.show()

    # Corner Plot
    burnin = int(0.3 * backend.iteration)
    flat_samples = backend.get_chain(discard=burnin, flat=True)
    corner.corner(flat_samples, labels=labels, show_titles=True,
                  quantiles=[0.16, 0.5, 0.84], title_fmt=".3f")
    plt.show()

def corner_plot_cdm(full_filename):
    """
    Aufruf: corner_plot_cdm('cdm_Test_DDO154_nw_16.h5')
    """
    filename = os.path.join("chains", full_filename)

    if not os.path.exists(filename):
        print(f"Datei nicht gefunden: {filename}")
        return

    backend = emcee.backends.HDFBackend(filename)
    samples = backend.get_chain()
    ndim = samples.shape[2]

    # Labels für CDM (normalerweise 2 Parameter)
    labels = ["log10(M200)", "c"]
    if ndim != len(labels):
        labels = [f"Param {i}" for i in range(ndim)]

    # --- NEU: Trace Plots für CDM ---
    fig, axes = plt.subplots(ndim, figsize=(8, 2 * ndim), sharex=True)

    # Falls ndim=1 (unwahrscheinlich), muss axes ein Array sein
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Steps")
    plt.tight_layout()
    plt.show()
    # --------------------------------

    # Corner Plot (bleibt gleich)
    burnin = int(0.3 * backend.iteration)
    flat_samples = backend.get_chain(discard=burnin, flat=True)

    print(f"Generating CDM corner plot for {ndim} parameters...")
    corner.corner(
        flat_samples,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt=".3f"
    )
    plt.show()
