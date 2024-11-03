import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_signal_direct_combined(
    bvalues_python,
    signal_allcmpts_python,
    signal_free_python,
    bvalues_matlab,
    signal_allcmpts_matlab,
    signal_free_matlab,
    title_str,
    experiment_colors,
    free_diffusion_colors,
):
    sns.set(style="darkgrid")

    # Plot Python data
    nsequence_python = bvalues_python.shape[1]
    for idir in range(signal_allcmpts_python.shape[-1]):
        plt.figure(figsize=(10, 6))
        for iseq in range(nsequence_python):
            yvec = np.real(signal_allcmpts_python[:, iseq, idir])
            bvec = bvalues_python[:, iseq]
            plt.plot(
                bvec,
                yvec,
                "o:",
                markersize=8,
                linewidth=2,
                color=experiment_colors["python"],
                label=f"Optimized Pytorch Seq {iseq + 1}, Direction {idir + 1}",
            )

            yvec = np.real(signal_allcmpts_matlab[:, iseq, idir])
            bvec = bvalues_matlab[:, iseq]
            plt.plot(
                bvec,
                yvec,
                "x:",
                markersize=8,
                linewidth=2,
                color=experiment_colors["matlab"],
                label=f"Reference Pytorch Seq {iseq + 1}, Direction {idir + 1}",
            )

            yvec_free = np.real(signal_free_python.ravel())
            plt.plot(
                bvalues_python.ravel(),
                yvec_free,
                "-",
                markersize=8,
                linewidth=2,
                color=free_diffusion_colors["python"],
                label="Optimzed Free diffusion",
            )

            yvec_free = np.real(signal_free_matlab.ravel())
            plt.plot(
                bvalues_matlab.ravel(),
                yvec_free,
                "-",
                markersize=8,
                linewidth=2,
                color=free_diffusion_colors["matlab"],
                label="Reference Free diffusion",
            )

        # Common plot adjustments
        plt.xlabel("B-value", fontsize=14)
        plt.ylabel("Signal", fontsize=14)
        plt.title(
            f"{title_str} - {nsequence_python} Sequences for Direction# {idir + 1}",
            fontsize=16,
        )
        plt.ylim([0, np.max([yvec_free.max(), yvec.max()]) * 1.1])
        x_min = min(np.min(bvalues_python), np.min(bvalues_matlab))
        x_max = max(np.max(bvalues_python), np.max(bvalues_matlab))
        plt.xlim([x_min, x_max])

        ticks = np.arange(x_min, x_max + 1000, 1000)
        plt.xticks(ticks, fontsize=12)
        plt.yticks(fontsize=12)

        plt.legend(fontsize=12, loc="upper right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        plt.show()

        # filename = f"{'/media/DATA_18_TB_1/prathamesh/i2500'}/plot_exp{iseq+1}_dir{idir+1}.png"
        # plt.savefig(filename)
        # plt.close()
