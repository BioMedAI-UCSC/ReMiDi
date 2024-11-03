import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_signal_direct_combined_normed(
    bvalues_python,
    signal_allcmpts_python,
    signal_free_python,
    bvalues_matlab,
    signal_allcmpts_matlab,
    signal_free_matlab,
    title_str,
    save_loc,
    experiment_colors,
    free_diffusion_colors,
    save_yes=False,
):
    sns.set(style="darkgrid")

    # Plot Python and MATLAB data
    nsequence_python = bvalues_python.shape[1]
    for idir in range(signal_allcmpts_python.shape[-1]):
        plt.figure(figsize=(10, 6))
        for iseq in range(nsequence_python):
            # Normalize Python signals
            yvec_python = np.real(signal_allcmpts_python[:, iseq, idir])
            max_yvec_python = np.max(yvec_python)
            normalized_yvec_python = yvec_python / max_yvec_python
            bvec_python = bvalues_python[:, iseq]
            plt.plot(
                bvec_python,
                normalized_yvec_python,
                "o:",
                markersize=8,
                linewidth=2,
                color=experiment_colors["python"],
                label=f"Optimized Pytorch Seq {iseq + 1}, Direction {idir + 1}",
            )

            # Normalize MATLAB signals
            yvec_matlab = np.real(signal_allcmpts_matlab[:, iseq, idir])
            max_yvec_matlab = np.max(yvec_matlab)
            normalized_yvec_matlab = yvec_matlab / max_yvec_matlab
            bvec_matlab = bvalues_matlab[:, iseq]
            plt.plot(
                bvec_matlab,
                normalized_yvec_matlab,
                "x:",
                markersize=8,
                linewidth=2,
                color=experiment_colors["matlab"],
                label=f"Reference Pytorch Seq {iseq + 1}, Direction {idir + 1}",
            )

            # Normalize and plot Python free diffusion signal
            yvec_free_python = np.real(signal_free_python.ravel())
            plt.plot(
                bvalues_python.ravel(),
                yvec_free_python / np.max(yvec_free_python),
                "-",
                markersize=8,
                linewidth=2,
                color=free_diffusion_colors["python"],
                label="Optimized Free diffusion",
            )

            # Normalize and plot MATLAB free diffusion signal
            yvec_free_matlab = np.real(signal_free_matlab.ravel())
            plt.plot(
                bvalues_matlab.ravel(),
                yvec_free_matlab / np.max(yvec_free_matlab),
                "-",
                markersize=8,
                linewidth=2,
                color=free_diffusion_colors["matlab"],
                label="Reference Free diffusion",
            )

        # Common plot adjustments
        plt.xlabel("B-value", fontsize=14)
        plt.ylabel("Normalized Signal", fontsize=14)
        plt.title(
            f"{title_str} - {nsequence_python} Sequences for Direction# {idir + 1}",
            fontsize=16,
        )
        plt.ylim(
            [0, 1.1]
        )  # Since normalization is applied, max is set to 1.1 for a little headroom
        x_min = min(np.min(bvalues_python), np.min(bvalues_matlab))
        x_max = max(np.max(bvalues_python), np.max(bvalues_matlab))
        plt.xlim([x_min, x_max])

        ticks = np.arange(x_min, x_max + 1000, 1000)
        plt.xticks(ticks, fontsize=12)
        plt.yticks(fontsize=12)

        plt.legend(fontsize=12, loc="upper right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        if save_yes:
            filename = f"{save_loc}/plot_exp{iseq+1}_dir{idir+1}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
