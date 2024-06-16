# Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach (CVPR24 Oral)

[![arXiv](https://img.shields.io/badge/arXiv-2404.00834-B31B1B.svg)](https://arxiv.org/abs/2404.00834)

## News :loudspeaker:

- **2024.04.06**: We plan to release our dataset and code no later than July.

## TODO list :pushpin:
- ~The release of synthetic event dataset of SDSD~ <br>
- The release of our collected SDE dataset <br>
- The release of our code

## SDE dataset :file_folder:
SED dataset contains 91 image+event paired sequences (43 indoor sequences and 48 outdoor sequences) captured with a DAVIS346 event camera which outputs RGB images and events with the resolution of 346*260.
For all collected sequences, 76 sequences are randomly selected for training, and 15 sequences are for testing. 

You can download the aglined dataset for experiments using the following link: baidu pan (uploading) and Onedrive (uploading).

*To reduce the size of files, we remove the split normal-light event streams and the whole normal-light event streams. If you need this additional data, pls send an email to us with your motivation and affiliation.*

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

You can download the processed event dataset for experiments using the following link: [baidu pan](https://pan.baidu.com/s/1b8ZXfHSzfWg0q0o4SgDcUQ?pwd=wrjv) (pwd: wrjv) and Onedrive (uploading).

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
