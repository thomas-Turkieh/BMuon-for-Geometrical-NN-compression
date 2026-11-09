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

`cache_n*/` folders hold the per-model evaluation points (n = number of held-out
structures). The figures are redrawn from them without a GPU.
