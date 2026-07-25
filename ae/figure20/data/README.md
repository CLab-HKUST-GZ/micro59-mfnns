# Figure 20 data

`figure20_area_power_breakdown.tsv` is a portable snapshot of the input to the
author's horizontal Figure 20 plotting script.

## Author-workspace source hashes

Recorded on 2026-07-25:

```text
df4e42efd21c02d9d4bf06365f241b78cbbb540e16e2473b251e08e87c640ba4  data.txt
9f36ac328bfd2782f3ef9494b1787be9aaeefe311629b063d0217e1d3fa0184f  plot_ap_breakdown_singlecol_horizontal.py
96b7dd9659d3e6fcbeae83f7bf94ec8c7405c70520fc02a1df506a00b47f2b2f  ap_breakdown_singlecol_horizontal.pdf
38e78079172960c3be9a934d4a556751098c548ddc38ea8eca78c7a64ec98f6a  ap_breakdown_singlecol_horizontal.png
```

The source directory was:

```text
/hpc2hdd/home/rmeng603/workspace/MFANNS/figure/evaluation/A_P_Breakdown
```

That absolute path is provenance only; reproducing this AE bundle does not
access it.

## Integrity

From this directory:

```bash
sha256sum -c SHA256SUMS
```

The PDF is excluded because Matplotlib embeds creation metadata. The frozen
TSV, normalized CSV, summary TSV, and PNG are inventoried.
