<div align="center">

# Towards Robust Event-guided Low-Light Image Enhancement: <br> A Large-Scale Real-World Event-Image Dataset and Novel Approach 

[**CVPR 2024 Oral & TPAMI 2025**]

<div>
    <a href="https://arxiv.org/abs/2404.00834" target="_blank">
        <img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square" alt="Paper">
    </a>
    <a href="https://vlislab22.github.io/eg-lowlight/" target="_blank">
        <img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page">
    </a>
    <a href="https://github.com/EthanLiang99/EvLight" target="_blank">
        <img src="https://img.shields.io/github/stars/EthanLiang99/EvLight?style=social" alt="GitHub Stars">
    </a>
</div>

<br>

</div>


## :loudspeaker: News

- **[2025.09.23]** :tada: Our extension paper **"EvLight++"** is now published in **IEEE TPAMI**! This work extends the original EvLight to **low-light video enhancement** with improved methodology and extensive applications (Source code is released).
- **[2024.12.12]** Normal-light event streams are released.
- **[2024.08.24]** Source code is released.
- **[2024.06.15]** SDE dataset and synthetic event dataset of SDSD are released.
- **[2024.04.06]** Dataset and code release plan announced.

## :pushpin: Roadmap & Status

- [x] Release of synthetic event dataset of SDSD
- [x] Release of our collected SDE dataset
- [x] Release of source code
- [x] Release of split normal-light event streams and the whole normal-light event streams

---

## :file_folder: Dataset Preparation

### 1. SDE Dataset (Real-World)
The **SDE dataset** contains **91** image+event paired sequences (43 indoor, 48 outdoor) captured with a DAVIS346.
* **Resolution**: 346 × 260
* **Split**: 76 training sequences, 15 testing sequences.

| Dataset Content | Baidu Netdisk | OneDrive | Password |
| :--- | :---: | :---: | :---: |
| **Aligned Dataset** | [Link](https://pan.baidu.com/s/1ad56IgSmCwDhorhwAwFlog?pwd=w7qe) | [Link](https://hkustgz-my.sharepoint.com/:f:/g/personal/gliang041_connect_hkust-gz_edu_cn/Ep_8Acz6cd1GjwtmEjAG0w8BkQsBWDjyHf9_56XSLTNLSw) | `w7qe` |
| **Normal-Light Events** | - | [Link](https://hkustgz-my.sharepoint.com/:f:/g/personal/gliang041_connect_hkust-gz_edu_cn/EtFRcH270mlNmXjgECTgrAgBsUA-qIRTuSzRtoNuGLa38g) | - |

> **Note**: We focus on the consistency between normal/low-light images. Consistency between event streams has not yet been fully verified.

<details>
<summary>Click to view SDE Directory Structure</summary>

```text
--indoor/outdoor 
├── test 
│   ├── pair1 
│   │   ├── low 
│   │   │   ├── xxx.png (low-light RGB frame) 
│   │   │   ├── xxx.npz (split low-light event streams) 
│   │   │   └── lowlight_event.npz (the whole low-light event stream) 
│   │   └── normal 
│   │       └── xxx.png (normal-light RGB frame) 
└── train 
    └── pair1 
        ├── low 
        │   ├── xxx.png 
        │   ├── xxx.npz 
        │   └── lowlight_event.npz 
        └── normal 
            └── xxx.png 
```
</details>

### 2. SDSD Dataset (Synthetic Events)
We incorporated events into the [SDSD](https://github.com/dvlab-research/SDSD) dataset using the [v2e](https://github.com/SensorsINI/v2e) simulator (resized to 346x260).

| Dataset Content | Baidu Netdisk | OneDrive | Password |
| :--- | :---: | :---: | :---: |
| **Processed Events** | [Link](https://pan.baidu.com/s/1b8ZXfHSzfWg0q0o4SgDcUQ?pwd=wrjv) | [Link](https://hkustgz-my.sharepoint.com/:f:/g/personal/gliang041_connect_hkust-gz_edu_cn/EsrS4qhMC_lFv3JgaGQ0nM8BG2GYHII_mBn2rYLhOpmN3g) | `wrjv` |

> :warning: **Notice**:
> 1. Please download the latest version (we fixed previous issues).
> 2. We recommend skipping the first/last 3 split event files due to sparse events caused by slow motion.

<details>
<summary>Click to view SDSD Directory Structure</summary>

```text
--indoor/outdoor 
├── test 
│   └── pair1 
│       ├── low (split low-light event streams for each RGB frame) 
│       └── low_event (whole synthetic low-light event stream) 
└── train 
    └── pair1 
        ├── low 
        └── low_event 
```
</details>

---

## :computer: Usage

### 1. Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pretrained Models
Download models from **[Baidu Pan](https://pan.baidu.com/s/1w9n1cl1Rom0GjVc3OOuFPw?pwd=8agv)** (pwd: `8agv`) or **[OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/gliang041_connect_hkust-gz_edu_cn/EiRk3QVTMttJmIIwvUoiwooB_0-tnlJ4yBVxzi4NxeLdmw)**.

Video-based checkpoints from **[Baidu Pan](https://pan.baidu.com/s/1CEznGOg__svhZD_N5SDpQA)** (pwd: `n1b7`) or **[OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/kchen879_connect_hkust-gz_edu_cn/IgBRTLwrw8GMRq3Qdnt2c67NAa5e1Yc-EDosBoE2o4NBmAM?e=5zpZTQ)**.

### 3. Training
1. Modify the dataset path in `options/train/xxx.yaml`.
2. Run the training script:
```bash
sh options/train/xxx.sh
```
> For video enhancement, use the corresponding `*_vid.sh` scripts.

### 4. Testing
1. Modify the model and dataset paths in `options/test/xxx.yaml`.
2. Run the test script:
```bash
sh options/test/xxx.sh
```
> For video enhancement, use the corresponding `*_vid.sh` scripts.

---

## :mortar_board: Citation

If this work is helpful for your research, please consider citing:

```bibtex
@ARTICLE{11192751,
  author={Chen, Kanghao and Liang, Guoqiang and Lu, Yunfan and Li, Hangyu and Wang, Lin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={EvLight++: Low-Light Video Enhancement With an Event Camera: A Large-Scale Real-World Dataset, Novel Method, and More},
  year={2026},
  volume={48},
  number={2},
  pages={1608-1625},
  keywords={Cameras;Videos;Semantic segmentation;Depth measurement;Feature extraction;Signal to noise ratio;Lighting;Semantics;Image color analysis;Training;Low light enhancement;high dynamic range;event camera;real-world dataset;downstream applications},
  doi={10.1109/TPAMI.2025.3617801}
}

@inproceedings{liang2024towards,
  title={Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach},
  author={Liang, Guoqiang and Chen, Kanghao and Li, Hangyu and Lu, Yunfan and Wang, Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23--33},
  year={2024}
}
```

## :heart: Acknowledgment
We thank the authors of [INR-Event-VSR](https://github.com/yunfanLu/INR-Event-VSR) and [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) for their open-source contributions.
