# Object-Aware Semantic Localization using Differential Evolution, PDFs, and KL Divergence

This repository contains the implementation associated with the research paper: **"Object-aware semantic mapping using probability density functions for indoor relocalization and path planning"** (Scientific Reports, 2026).

The project implements a robust robot localization framework powered by Differential Evolution (DE), Probability Density Functions (PDFs), and Kullback-Leibler (KL) divergence.

---

## 📁 Repository Structure

1. **`geometricLoc/`**
   * **Description:** Implements localization via Differential Evolution using a purely local geometric map representation.

2. **`semanticLoc/`**
   * **Description:** Extends the localization framework to semantic mapping, matching local and global point clouds labeled with color information using DE, PDFs, and KL divergence.

---

## 📄 Citation

If you use this code or work in your research, projects, or teaching, please cite it as:

```bibtex
@article{mora2026object,
  title     = {Object-aware semantic mapping using probability density functions for indoor relocalization and path planning},
  author    = {Mora, Alicia and Mendez, Alberto and Moreno, Luis and Barber, Ramon},
  journal   = {Scientific Reports},
  volume    = {16},
  number    = {1},
  pages     = {9450},
  year      = {2026},
  publisher = {Nature Publishing Group UK, London}
}
