# Results: figures and tables

One folder per experiment. The folder names match [`../runs/`](../runs/), where the training
logs are, and [`../models/`](../models/), where the trained models are. Errors are mean absolute
errors on held-out MPtrj structures: forces in meV/Å, energy in meV/atom. Lower is better.

| Folder | Content | Main files |
|---|---|---|
| [`2026-10-13_nequix-mp1_inference-speed-rtx3090`](2026-10-13_nequix-mp1_inference-speed-rtx3090/) | Inference time of Nequix-MP-1 on an RTX 3090 when channels are removed, per block and irrep (§6.2). | `speed_3090*.csv/json` |
| [`2026-10-13_nequix-mp1_channel-fragility`](2026-10-13_nequix-mp1_channel-fragility/) | Accuracy of the published Nequix-MP-1 when channels are removed without fine-tuning (§6.1). | `fragility_mp1_global.csv`, `fragility_mp1_pergroup_log.txt` |
| [`2026-10-13_full-mptrj_dense-training_bmuon-vs-uniform`](2026-10-13_full-mptrj_dense-training_bmuon-vs-uniform/) | Training curves of the October runs: BMuon with a 30 % budget against a small uniform model of the same cost. | `training_curves.png` (English), `training_curves_fr.png` (French), `val_errors_per_epoch.csv` |
| [`2026-11-03_full-mptrj_post-training-pruning`](2026-11-03_full-mptrj_post-training-pruning/) | Classic pruning: the models trained on full MPtrj (October runs and the published Nequix-MP-1) are pruned after training, without fine-tuning, by magnitude or by the BMuon allocation μ. | `pruning_forces.png`, `pruning_energy.png`, `table_forces.md`, `fig2_meeting*.png`, `fig2*.png`, `fig2_points.csv` |
| [`2026-11-03_20pct-mptrj_pruning-methods-comparison`](2026-11-03_20pct-mptrj_pruning-methods-comparison/) | The paper's Figure 2 for Nequix at reduced scale (20 % of MPtrj, 25 epochs): post-training pruning of every method, plus the gradual variants. | `fig2_final.png` (forces), `fig2_final_energy.png`, `fig2_final_points.csv`, `STATUS.md` |
| [`2026-11-06_full-mptrj_gradual-pruning`](2026-11-06_full-mptrj_gradual-pruning/) | Gradual pruning during training against classic pruning of the trained models, and against small models trained from scratch (stars), on full MPtrj. | `gradual_vs_classic.png` (forces), `gradual_vs_classic_energy.png`, `table.md`, `points.csv`, `training_curves.png` |
| [`2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size`](2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size/) | Models with 1.25× to 3× wider tensor products, gradually pruned (BM-μ) to the size of Nequix-MP-1, against the dense Nequix-MP-1 (no pruning). Training curves (partial while x1.25 and x3 train; redraw with `python -m bmuon.experiments.wide_fig`), and the full comparison of the final 1.5× model with the dense Nequix-MP-1: errors on the whole validation split with 95 % bootstrap confidence intervals, inference time and training cost measured on the same GPU (`python -m bmuon.experiments.equal_size_bench`, then `equal_size_fig`). | `equal_size_comparison.png/.pdf/.json`, `training_curves_partial.png`, `val_errors_per_epoch.csv` |
| [`2026-11-22_full-mptrj_x1.5-pruned-architecture-from-scratch`](2026-11-22_full-mptrj_x1.5-pruned-architecture-from-scratch/) | The architecture kept by the 1.5× wide model, retrained from scratch, against the dense Nequix-MP-1 (uniform architecture, same cost): training curves and per-epoch difference. Partial while the run trains; redraw with `python -m bmuon.experiments.scratch_vs_dense_fig`. | `scratch_vs_dense_curves.png/.pdf` |
| [`2026-11-24_full-mptrj_meeting-pre-read`](2026-11-24_full-mptrj_meeting-pre-read/) | Two-page meeting pre-read (latest results of the gradual pruning of wider models and of the from-scratch run), compiled from `pre-read.tex`; figures from `python -m bmuon.experiments.preread_figs`. | `pre-read_2026-11-24.pdf`, `fig1`–`fig4` (.png/.pdf), `summary.json` |

`cache_n*/` folders hold the per-model evaluation points (n = number of held-out
structures). The figures are redrawn from them without a GPU.
