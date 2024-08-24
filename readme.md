# Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach (CVPR24 Oral)

[![arXiv](https://img.shields.io/badge/arXiv-2404.00834-B31B1B.svg)](https://arxiv.org/abs/2404.00834)

## News :loudspeaker:

- **2024.04.06**: We plan to release our dataset and code no later than July.
- **2024.06.15**: Our dataset and synthetic event dataset of SDSD are released :)
- **2024.08.24**: Our code is released :)

## TODO list :pushpin:
- ~The release of synthetic event dataset of SDSD~ <br>
- ~The release of our collected SDE dataset~ <br>
- ~The release of our code~ <br>
- The release of split normal-light event streams and the whole normal-light event streams

## Usage :computer:
**Dependency**

```
pip install -r requirements.txt
```

**Test**

- Download pretrained models ([baidu pan](https://pan.baidu.com/s/1w9n1cl1Rom0GjVc3OOuFPw?pwd=8agv) (pwd: 8agv) and [Onedrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/gliang041_connect_hkust-gz_edu_cn/EiRk3QVTMttJmIIwvUoiwooB_0-tnlJ4yBVxzi4NxeLdmw)) 

- Change the path to the models and dataset in options/test/xxx.yaml .

```
sh options/test/xxx.sh
```

**Train**

- Change the path to dataset in options/train/xxx.yaml .

```
sh options/train/xxx.sh
```


## SDE dataset :file_folder:
SED dataset contains 91 image+event paired sequences (43 indoor sequences and 48 outdoor sequences) captured with a DAVIS346 event camera which outputs RGB images and events with the resolution of 346*260.
For all collected sequences, 76 sequences are randomly selected for training, and 15 sequences are for testing. 

You can download the aglined dataset for experiments using the following link: [baidu pan](https://pan.baidu.com/s/1ad56IgSmCwDhorhwAwFlog?pwd=w7qe) (pwd: w7qe) and [Onedrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/gliang041_connect_hkust-gz_edu_cn/Ep_8Acz6cd1GjwtmEjAG0w8BkQsBWDjyHf9_56XSLTNLSw).

The arrangement of the dataset is
```
--indoor/outdoor 
| 
----test 
|   | 
|   ----pair1 
|       | 
|       ----low 
|       |   | 
|       |   ----xxx.png (low-light RGB frame) 
|       |   ----xxx.npz (split low-light event streams) 
|       |   ----lowligt_event.npz (the whole low-light event stream) 
|       | 
|       ----normal 
|           | 
|           ----xxx.png (normal-light RGB frame) 
| 
----train 
    | 
    ----pair1 
        | 
        ----low 
        |   | 
        |   ----xxx.png (low-light RGB frame) 
        |   ----xxx.npz (split low-light event streams) 
        |   ----lowligt_event.npz (the whole low-light event stream) 
        | 
        ----normal 
            | 
            ----xxx.png (normal-light RGB frame) 
```

## SDSD dataset :file_folder:
Unlike the original configuration in [SDSD](https://github.com/dvlab-research/SDSD), our project incorporates events as an additional input. We have provided the processed event data for this purpose. To prepare the data, we downsampled the original videos to the resolution of the DAVIS346 event camera (346x260) and inputted these resized images into the [v2e](https://github.com/SensorsINI/v2e) event simulator. This simulator uses its default model to synthesize noisy event streams.

You can download the processed event dataset for experiments using the following link: [baidu pan](https://pan.baidu.com/s/1b8ZXfHSzfWg0q0o4SgDcUQ?pwd=wrjv) (pwd: wrjv) and [Onedrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/gliang041_connect_hkust-gz_edu_cn/EsrS4qhMC_lFv3JgaGQ0nM8BG2GYHII_mBn2rYLhOpmN3g). * We fix the problem in the SDSD out dataset, pls download the latest version!*

*Due to slow motion at the beginning and end of videos, the generated events may be sparse. Therefore, we recommend skipping the first three and last three split event files.*

The arrangement of the dataset is
```
--indoor/outdoor 
| 
----test 
|   | 
|   ----pair1 
|       | 
|       ----low (the split low-light event streams for each RGB frame) 
|       ----low_event (the whole synthetic low-light event stream) 
| 
----train 
    | 
    ----pair1 
        | 
        ----low (the split low-light event streams for each RGB frame) 
        ----low_event (the whole synthetic low-light event stream) 
```


## Citations :mortar_board:
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@inproceedings{liang2024towards,
  title={Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach},
  author={Liang, Guoqiang and Chen, Kanghao and Li, Hangyu and Lu, Yunfan and Wang, Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23--33},
  year={2024}
}
```


## Acknowledgment:
Thanks to these open source projects: [INR-Event-VSR](https://github.com/yunfanLu/INR-Event-VSR) and [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer).