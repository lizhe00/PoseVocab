
<div align="center">

# <b>PoseVocab</b>: Learning Joint-structured Pose Embeddings for Human Avatar Modeling

<h2>SIGGRAPH 2023</h2>

[Zhe Li](https://lizhe00.github.io/), [Zerong Zheng](https://zhengzerong.github.io/), Yuxiao Liu, Boyao Zhou, [Yebin Liu](https://www.liuyebin.com)

Tsinghua Univserity

### [Projectpage](https://lizhe00.github.io/projects/posevocab/) · [Paper](https://arxiv.org/pdf/2304.13006.pdf) · [Video](https://youtu.be/L-kg74A6yNc)

</div>

## Introduction
We propose PoseVocab, a novel pose encoding method that encodes dynamic human appearances under various poses for human avatar modeling.

https://user-images.githubusercontent.com/61936670/243704320-991c017f-16aa-4bda-814c-579a4a7be784.mp4


## Installation
Clone this repo, then run the following scripts.
``` 
cd ./utils/posevocab_custom_ops
python setup.py install
cd ../..
```

## SMPL-X & Pretrained Models
- Download [SMPL-X files](https://smpl-x.is.tue.mpg.de/download.php), place pkl files to ```./smpl_files/smplx```.
- Download [pretrained models](https://drive.google.com/file/d/10nqtueMuOHKNz0phDc2mQCh4cJsBj97Q/view?usp=sharing), unzip it to ```./pretrained_models```.

## Run on THuman4.0 Dataset
### Dataset Preparation
- Download [THuman4.0 dataset](https://github.com/ZhengZerong/THUman4.0-Dataset). Let's take "subject00" as an example, and denote the root data directory as ```SUBJECT00_DIR```.
- Specify the data directory and training frame list in ```gen_data/main_preprocess.py```, then run the following scripts.
```
cd ./gen_data
python main_preprocess.py
cd ..
```

### Training
*Note: In the first training stage, our method reconstructs depth maps for the depth-guided sampling in the next stages.
If you want to skip the first stage, you can download our provided depth maps from [this link](https://drive.google.com/file/d/1rEaaf-ayXXRUEQFJ2fUut0-xRKW_r2K1/view?usp=sharing), unzip it to ```SUBJECT00_DIR/depths```, and directly run ```python main.py -c configs/subject00.yaml -m train``` until the network converges.*

- *Stage 1: training to obtain depth maps.*
Set ```end_epoch``` in [configs/subject00.yaml#L15](configs/subject00.yaml#L15) to 10.
```
python main.py -c configs/subject00.yaml -m train
```

- *Stage 2: render depth maps.*
```
python main.py -c configs/subject00.yaml -m render_depth_sequences
```

- *Stage 3: continue training.* Set ```start_epoch``` in [configs/subject00.yaml#L14](configs/subject00.yaml#L14) to 11, and ```prev_ckpt``` in [configs/subject00.yaml#L12](configs/subject00.yaml#L12) to ```./results/subject00/epoch_latest```.
```
python main.py -c configs/subject00.yaml -m train
```

### Testing
Download testing poses from [this link](https://drive.google.com/file/d/1LfvqDYz3k_WGDi2m9C0isfkRdDcnJsLH/view?usp=sharing), unzip them to somewhere, denoted as ```TESTING_POSE_DIR```.
- Specify ```prev_ckpt``` in [configs/subject00.yaml#L78](configs/subject00.yaml#L78) as the pretrained model ```./pretrained_models/subject00``` or the trained one by yourself.
- Specify ```data_path``` in [configs/subject00.yaml#L60](configs/subject00.yaml#L60) as the testing pose path, e.g., ```TESTING_POSE_DIR/thuman4/pose_01.npz```.
- Run the following script.
```
python main.py -c configs/subject00.yaml -m test
```
- The output results can be found in ```./test_results/subject00```.

## License
MIT License. SMPL-X related files are subject to the license of [SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html).

## Citation
If you find our code or paper is useful to your research, please consider citing:
```bibtex
@inproceedings{li2023posevocab,
  title={PoseVocab: Learning Joint-structured Pose Embeddings for Human Avatar Modeling},
  author={Li, Zhe and Zheng, Zerong and Liu, Yuxiao and Zhou, Boyao and Liu, Yebin},
  booktitle={ACM SIGGRAPH Conference Proceedings},
  year={2023}
}
```
