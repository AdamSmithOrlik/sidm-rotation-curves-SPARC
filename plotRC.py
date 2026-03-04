import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from readData import get_galaxy_ids, get_rc_data


# plot the rotation curves for each galaxy
def main():
    savepath = os.getcwd() + "/plots/rc_data/"

    # get the galaxy names from the filenames
    galaxy_IDs = get_galaxy_ids()

    # for galaxy in galaxy_IDs:
    #     df, units_df, distance = get_rc_data(galaxy)

    #     vc_baryons = np.sqrt(df["Vdisk"] ** 2 + df["Vbul"] ** 2 + df["Vgas"] ** 2)

    #     plt.figure(figsize=(6, 4))
    #     plt.scatter(
    #         df["Rad"], vc_baryons, label="Baryon Total", color="k", s=10, zorder=0
    #     )
    #     plt.scatter(df["Rad"], df["Vdisk"], label="Disk", color="C0", s=8)
    #     plt.scatter(df["Rad"], df["Vbul"], label="Bulge", color="C3", s=8)
    #     plt.scatter(df["Rad"], np.abs(df["Vgas"]), label="Gas", color="C1", s=8)

    #     plt.xlabel(f"Radius (kpc)")
    #     plt.ylabel(f"Velocity (km/s)")
    #     plt.title(f"Galaxy {galaxy} at Distance {distance} Mpc")
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(savepath + f"{galaxy}_RC.png")
    #     plt.close()

    results = []
    for igx, galaxy in enumerate(galaxy_IDs):
        df, units_df, distance = get_rc_data(galaxy)

        vdisk = df["Vdisk"].values.sum(axis=0)
        vbul = df["Vbul"].values.sum(axis=0)
        vgas = np.abs(df["Vgas"].values).sum(axis=0)
        vc_total = np.sqrt(vdisk**2 + vbul**2 + vgas**2).sum(axis=0)

        vdisk_ratio = vdisk / vc_total
        vbul_ratio = vbul / vc_total
        vgas_ratio = vgas / vc_total

        results.append(
            {
                "Galaxy": galaxy,
                "igx": igx,
                "Vdisk Ratio": vdisk_ratio,
                "Vbul Ratio": vbul_ratio,
                "Vgas Ratio": vgas_ratio,
            }
        )

        # plt.figure(figsize=(6, 4))
        # plt.bar(
        #     ["Disk", "Bulge", "Gas"],
        #     [vdisk_ratio, vbul_ratio, vgas_ratio],
        #     color=["C0", "C3", "C1"],
        # )
        # plt.ylabel("Velocity Ratio")
        # plt.title(f"Galaxy {galaxy} Velocity Ratios")
        # plt.savefig(savepath + f"{galaxy}_velocity_ratios.png")
        # plt.close()

    results_df = pd.DataFrame(
        results, columns=["Galaxy", "igx", "Vdisk Ratio", "Vbul Ratio", "Vgas Ratio"]
    )

    plt.figure(figsize=(6, 4))

    plt.scatter(
        results_df["igx"],
        results_df["Vdisk Ratio"] / results_df["Vgas Ratio"],
        label="Disk/Gas",
        color="k",
        s=10,
    )
    plt.scatter(
        results_df["igx"],
        results_df["Vdisk Ratio"] / results_df["Vbul Ratio"],
        label="Disk/Bulge",
        color="grey",
        s=10,
    )

    plt.axhline(1, color="k", linestyle="--", label="Equal Contribution")
    plt.xlabel("Galaxy Index (igx)")
    plt.ylabel("Velocity Ratio (Disk/Gas and Disk/Bulge)")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(savepath + "velocity_comparison.png")
    plt.close()

    # plt.figure(figsize=(6, 4))
    # plt.scatter(
    #     results_df["igx"],
    #     results_df["igx"] * results_df["Vdisk Ratio"],
    #     label="Disk",
    #     color="C0",
    #     s=10,
    # )
    # plt.scatter(
    #     results_df["igx"],
    #     results_df["igx"] * results_df["Vbul Ratio"],
    #     label="Bulge",
    #     color="C3",
    #     s=10,
    # )
    # plt.scatter(
    #     results_df["igx"],
    #     results_df["igx"] * results_df["Vgas Ratio"],
    #     label="Gas",
    #     color="C1",
    #     s=10,
    # )
    # plt.plot(
    #     results_df["igx"], results_df["igx"], label="Total", color="k", linestyle="--"
    # )
    # plt.xlabel("Galaxy Index (igx)")
    # plt.ylabel("Velocity Ratio * Galaxy Index")
    # plt.legend(loc="upper left")

    # plt.tight_layout()
    # plt.savefig(savepath + "velocity_ratios_comparison.png")
    # plt.close()


if __name__ == "__main__":
    main()
