"""
Per-residue S2 comparison across 9 BMRB entries, three traces each:

  algorithm='wishart' : makeshift.rci.RCI(algorithm="wishart")'s own S2
                         column -- the published Berjanskii & Wishart 2005
                         relation, S2 = 1 - 0.5*ln(1+10*RCI), applied to
                         the RCI values makeshift computes (an exact port
                         of rci_v_1c.py, machine-precision validated --
                         see tests/test_rci_regression.py). Conventional
                         S2 sense (rigid -> ~1, flexible -> lower).
  wrapped TALOS-N      : the real compiled TALOS-N binary's own
                         predS2.tab output, via makeshift.talosn.TalosN.
  algorithm='talosn'   : makeshift.rci.RCI(algorithm="talosn"), a
                         from-source port of TALOS-N's bundled
                         RCI.cpp/TALOS.cpp (a simpler,
                         independently-drifted reimplementation of
                         Wishart's RCI, not rci_v_1c.py) -- see
                         makeshift/rci/_talosn.py. Already validated
                         against the real binary on these same 9 entries;
                         this figure is the visual version of that
                         validation, and should track "wrapped TALOS-N"
                         closely while algorithm='wishart' is expected to
                         differ, since it's a different algorithm
                         (rci_v_1c.py vs. TALOS-N's own RCI.cpp).

Fetches chemical shifts live via ChemicalShifts.from_bmrb(keep_download=True)
-- the downloaded *.str files are gitignored, so this isn't run in CI.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import makeshift as ms
from makeshift.rci import RCI
from makeshift.talosn import TalosN

TALOSN_DATA_DIR = Path.home() / "talosn_data"
OUTPUT_FILE = Path(__file__).parent / "rci_algorithm_validation.png"

BMRB_IDS = [11080, 15451, 15490, 15521, 15581, 15763, 15959, 52018, 5991]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, bmrb_id in zip(axes, BMRB_IDS):
        cs = ms.ChemicalShifts.from_bmrb(bmrb_id, keep_download=True)

        wishart = RCI.calc(cs, algorithm="wishart")
        ax.plot(wishart.results["Seq_ID"], wishart.results["S2"],
                label="algorithm='wishart'", color="steelblue", lw=1.3)

        talosn_port = RCI.calc(cs, algorithm="talosn")
        finite = talosn_port.results["S2"] < 2.0  # drop the 9999.0 sentinel
        ax.plot(talosn_port.results.loc[finite, "Seq_ID"], talosn_port.results.loc[finite, "S2"],
                label="algorithm='talosn'", color="firebrick", lw=1.3, ls="-.")

        tn = TalosN.from_entry(cs.entry, data_dir=TALOSN_DATA_DIR)
        tn.run(auto_install=True)
        talosn_s2 = tn.order_parameters
        if talosn_s2 is not None and not talosn_s2.empty:
            talosn_s2 = talosn_s2.rename(columns={"RESID": "Seq_ID"})
            talosn_s2 = talosn_s2[talosn_s2["S2"] < 2.0]
            ax.plot(talosn_s2["Seq_ID"], talosn_s2["S2"],
                    label="wrapped TALOS-N", color="0.3", lw=1.3, ls=":")

        ax.set_title(f"BMRB {bmrb_id}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("residue")
        ax.set_ylabel("S2")
        print(f"{bmrb_id}: done")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT_FILE, dpi=200)
    print(f"saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
