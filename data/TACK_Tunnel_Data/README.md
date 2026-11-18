---
language:
- en
pretty_name: TTD
---

# TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels

Tunnels are essential elements of transportation infrastructure, but are increasingly affected by ageing and deterioration mechanisms such as cracking. Regular inspections are required to ensure their safety, yet traditional manual procedures are time-consuming, subjective, and costly. Recent advances in mobile mapping systems and Deep Learning (DL) enable automated visual inspections. However, their effectiveness is limited by the scarcity of tunnel datasets. This is the official repository of a new publicly available dataset containing annotated images of three different tunnel linings, capturing typical defects: cracks, leaching, and water infiltration. The dataset is designed to support supervised, semi-supervised, and unsupervised DL methods for defect detection and segmentation. Its diversity in texture and construction techniques also enables investigation of model generalization and transferability across tunnel types. By addressing the critical lack of domain-specific data, this dataset contributes to advancing automated tunnel inspection and promoting safer, more efficient infrastructure maintenance strategies. 

## Dataset Description

Data are collected from three tunnels (Tunnel A, Tunnel B, Tunnel C) using a mobile mapping system equipped with high-resolution cameras. 
The three tunnels exhibit different surfaces due to the construction methods used, as visible in Figure 1.

<p align="left">
  <img src="figures/figure1.png" width="600"/>
</p>
<p align="left">
  <em>Figure 1: Example of high-resolution images depicting the surface of the three tunnels.</em>
</p>

The dataset includes three classes of defects from each tunnel. In total, 785 images with cracks, 197 images with water and 316 images with leaching (Table 1). 

<p align="left">
  <img src="figures/table1.png" width="600"/>
</p>
<p align="left">
  <em>Table 1: Description of the complete dataset, divided into three tunnel types and three classes.</em>
</p>

For each image, a segmentation mask is provided. Masks were generated manually using the SuperAnnotate platform.

For any additional detail and technical validation of the dataset, please refer to the preprint https://arxiv.org/abs/2512.14477.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{sjölander2025tacktunneldatattd,
      title={TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels}, 
      author={Andreas Sjölander and Valeria Belloni and Robel Fekadu and Andrea Nascetti},
      year={2025},
      eprint={2512.14477},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.14477}, 
}