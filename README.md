# Breast Cancer Subtype Analysis with Self-Organizing Maps

Unsupervised categorization of breast cancer patients using Self-Organizing Maps (SOMs) trained on multi-omics METABRIC data.

https://github.com/Tectonius/NML\_final\_project

## Overview

This project trains a SOM on integrated clinical, mutational, copy number, and gene expression data from the METABRIC breast cancer cohort to discover patient subgroups in an unsupervised manner. The trained SOM lattice is then visualized with overlays of known molecular subtypes (PAM50 + Claudin-low) and clinical variables to interpret the learned topology.

## Data

Source: **cBioPortal METABRIC study** 

After preprocessing, the final training input contains **1,894 samples × 522 features**:

- **Clinical features (11):** Age at Diagnosis, Tumor Size, Lymph Nodes Positive, Nottingham Prognostic Index, Mutation Count, Cancer Type Detailed, Cellularity, Histologic Grade, Tumor Stage, Inferred Menopausal State, Primary Tumor Laterality
- **Somatic mutations (173 genes):** Binary 0/1
- **Copy number alterations (172 genes):** Integer from −2 to +2
- **mRNA expression z-scores (173 genes):** Continuous

Response variables and subtype labels are tracked but not used as SOM inputs.

## Final SOM Configuration

- **Lattice:** 20 × 20
- **Training steps:** 1,000,000
- **Learning rate:** Piecewise linear schedule (0.8 - 0.4 - 0.1 - 0.03 - 0.01)
- **Initial neighborhood width:** 10.0

## Usage

### Training

```bash
python python_scripts/train_metabric_som.py \
  --lattice-rows 20 --lattice-cols 20 \
  --training-steps 1000000 \
  --lr-schedule "0:0.8,0.05:0.4,0.15:0.1,0.4:0.03,1.0:0.01" \
  --initial-width 10.0 --seed 42
```

### Visualization

Generate mU-matrix with PAM50 subtype overlay:
```bash
python python_scripts/plot_pam50_mu_matrix.py {run directory}
```

Generate response graphs with PAM50 overlay:
```bash
python python_scripts/plot_pam50_overlay_response_graphs.py {run directory}
```

Generate chromosome-arm summed heatmaps with PAM50 overlay:
```bash
python python_scripts/plot_location_summed_pam50_overlay.py {run directory}
```

## Packages

- NumPy, Pandas, Matplotlib
