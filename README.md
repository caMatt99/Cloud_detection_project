# Cloud Detection — Ceilometer Backscatter Profiles

Progetto di benchmarking per il rilevamento di nuvole da dati lidar ceilometer.

## Dataset
Dataset pubblico: https://zenodo.org/records/10616434

## Modelli testati
- Baseline: ResNet50 (89.57% accuracy — da paper)
- Nuovo: ConvNeXt-Base

## Struttura
- `notebooks/` — analisi esplorativa e training
- `src/` — codice modulare riusabile
- `configs/` — iperparametri
- `results/` — grafici e metriche
