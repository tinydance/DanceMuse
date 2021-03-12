[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
## Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning
**\*\*\*\*\*\*\*\*\* June 19, 2020 \*\*\*\*\*\*\*\*\*** <br>
The code and data are going through the internal review and will be released later!

**\*\*\*\*\*\*\*\*\* August 26, 2020 \*\*\*\*\*\*\*\*\*** <br>
The dataset is still going through the internal review, please wait.

**\*\*\*\*\*\*\*\*\* September 7, 2020 \*\*\*\*\*\*\*\*\*** <br>
The code & pose data are released!


### Introduction
This repo is the PyTorch implementation of "Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning". Our proposed approach significantly outperforms the existing methods and achieves the state-of-art performance in extensive experiments. It can generate creative long dance sequences, e.g., about <strong>1-minute length under 15 FPS</strong>, from the input music clips, which are smooth, natural-looking, diverse, style-consistent and beat-matching with the music. This techique can be used to drive various 3D character models by 3D reconstruction and animation driving, and has the great potential for the virtual ads video generation on social medias like TikTok.

### Paper 
Ruozi Huang*, [Huang Hu*](https://stonyhu.github.io/), [Wei Wu](https://sites.google.com/view/wei-wu-homepage), [Kei Sawada](http://www.sp.nitech.ac.jp/~swdkei/index.html), [Mi Zhang](http://homepage.fudan.edu.cn/zhangmi/en) and Daxin Jiang <br/>
Dance Revolution: Long Sequence Dance Generation with Music via Curriculum Learning <br/>
[[arXiv]](https://arxiv.org/pdf/2006.06119.pdf) [[YouTube]](https://www.youtube.com/watch?v=P6yhfv3vpDI) [Project]

### Requirements
- Python 3.7
- PyTorch 0.4.1

### Dataset and Installation
- We released the dance pose data and the corresponding audio data into [here](https://drive.google.com/file/d/1FGGF7P_gR8ssfewhVogskvDyu6Gb6Dr8/view?usp=sharing). The pose sequences are extracted from original dance videos with 30FPS while the audio data is m4a format. Note that, we develope a simple interpolation alogrithm to find missing keyjoints to reduce the noise in the pose data, which is introduced by the imperfect extraction of OpenPose.

- If you plan to train the model with your own dance data, please install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for the human pose extraction.

### Generated Example Videos
- Ballet style
<p align='center'>
  <img src='imgs/ballet-1.gif' width='400'/>
  <img src='imgs/ballet-2.gif' width='400'/>
</p>

- Hiphop style
<p align='center'>
  <img src='imgs/hiphop-1.gif' width='400'/>
  <img src='imgs/hiphop-2.gif' width='400'/>
</p>

- Japanese Pop style
<p align='center'>
  <img src='imgs/pop-1.gif' width='400'/>
  <img src='imgs/pop-2.gif' width='400'/>
</p>

- Photo-Realisitc Videos by [vid2vid](https://github.com/NVIDIA/vid2vid) </br>
We map the generated skeleton dances to the photo-realistic videos by [vid2vid](https://github.com/NVIDIA/vid2vid). Specifically, We record a random dance video of a team memebr to train the vid2vid model. Then we generate photo-realistic videos by feeding the generated skeleton dances to the trained vid2vid model. Note that, our team member has authorized us the usage of her portrait in following demos. 
<p align='center'>
  <img src='imgs/skeleton-1.gif' width='428'/>
  <img src='imgs/kazuna-crop-1.gif' width='400'/>
  <img src='imgs/skeleton-2.gif' width='429'/>
  <img src='imgs/kazuna-crop-2.gif' width='400'/>
</p>

- Driving 3D model by applying [3D human pose estimation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ci_Optimizing_Network_Structure_for_3D_Human_Pose_Estimation_ICCV_2019_paper.pdf)  and Unity animation to generated skeleton dances.


### Citation
If you find this work useful for your research, please cite the following paper :-)
```
@article{huang2020dance,
  title={Dance Revolution: Long Sequence Dance Generation with Music via Curriculum Learning},
  author={Huang, Ruozi and Hu, Huang and Wu, Wei and Sawada, Kei and Zhang, Mi},
  journal={arXiv preprint arXiv:2006.06119},
  year={2020}
}
```
