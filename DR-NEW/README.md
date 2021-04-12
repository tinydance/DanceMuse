[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
## Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning
**\*\*\*\*\*\*\*\*\* June 19, 2020 \*\*\*\*\*\*\*\*\*** <br>
The code and data are going through the internal review and will be released later!

**\*\*\*\*\*\*\*\*\* August 26, 2020 \*\*\*\*\*\*\*\*\*** <br>
The dataset is still going through the internal review, please wait.

**\*\*\*\*\*\*\*\*\* September 7, 2020 \*\*\*\*\*\*\*\*\*** <br>
The code & pose data are released!

**\*\*\*\*\*\*\*\*\* April 4, 2021 \*\*\*\*\*\*\*\*\*** <br>
Two versions of codebase are released. Have a try to train your AI dancer!


### Introduction
This repo is the PyTorch implementation of "Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning". Our proposed approach significantly outperforms the existing SOTAs in extensive experiments, including automatic metrics and human judgements. It can generate creative long dance sequences, e.g., about <strong>1-minute length under 15 FPS</strong>, from the input music clips, which are smooth, natural-looking, diverse, style-consistent and beat-matching with the music from test set. With the help of 3D human pose estimation and 3D animation driving, this technique can be used to drive various 3D character models such as the 3D model of Hatsune Miku (very popular virtual character in Japan), and has the great potential for the virtual advertisement video generation.

### Paper 
Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning. <strong>ICLR 2021</strong>. <br/>
Ruozi Huang*, [Huang Hu*](https://stonyhu.github.io/), [Wei Wu](https://sites.google.com/view/wei-wu-homepage), [Kei Sawada](http://www.sp.nitech.ac.jp/~swdkei/index.html), [Mi Zhang](http://homepage.fudan.edu.cn/zhangmi/en) and [Daxin Jiang](https://www.microsoft.com/en-us/research/people/djiang/). <br/> 
[[Paper]](https://openreview.net/pdf?id=xGZG2kS5bFk) [[YouTube]](https://youtu.be/lmE20MEheZ8) [Project]

### Requirements
- Python 3.7
- PyTorch 1.6.0
- ffmpeg

### Dataset and Installation
- We released the dance pose data and the corresponding audio data into [[Google Drive]](https://drive.google.com/file/d/1FGGF7P_gR8ssfewhVogskvDyu6Gb6Dr8/view?usp=sharing). 
Please put the downloaded `data/` into the project directory `DanceRevolution/` and run `prepro.py` that will generate the training data directory `data/train_1min` and test data directory `data/test_1min`. The pose sequences are extracted from the collected dance videos with original 30FPS while the audio data is m4a format. After the generation is finished, you can run `interpolate_to30fps.py` to increase the 15FPS to 30FPS to produce visually smoother dance videos for generated results.

- If you plan to train the model with your own dance data (2D), please install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for the human pose extraction. After that, please follow the hierarchical structure of directory `data/` to place your own extracted data and run `prepro.py` to generate the training data and test data. Note that, we develope a `interpolate_missing_keyjoints.py` script to find the missing keyjoints to reduce the noise in the pose data, which is introduced by the imperfect extraction of OpenPose.

- Due to the lack of 3D dance data in hand, we did not test our approach on 3D data. Recently, there is a wonderful paper conducted by Li et al., "Learn to Dance with AIST++: Music Conditioned 3D Dance Generation"(https://arxiv.org/abs/2101.08779), on music-conditioned 3D dance generation, which releases a large-scale 3D human dance motion dataset, AIST++. Just heard from the authors of this paper that our approach also performs well on their released 3D dataset and add our approach as one of compared baselines in their work!


### Training Issues
We released two versions of codebases that have passed the test. In V1 version, the local self-attention module is implemented base on the [longformer](https://github.com/allenai/longformer) that provides the custom CUDA kernel to accelerate the training speed and save GPU memory for long sequence inputs. While V2 version just implements the local self-attention module via the naive PyTorch implementation, i.e., the attention mask operations. In practice, we found the performance of V2 is more stable and recommend to use V2 version. Here are some training tricks that may be helpful for you:
- Small batch sizes, such as 32 and 16, would help model to converge better and the model usually converges well at around the 3000-th epoch. It takes about 3 days to train the model well under these settings.
- Increasing sliding window size of local self-attention is beneficial to the more stable performance while the cost (e.g., training time and GPU memory usage) would become high. This point has been empirically justified in the ablation study of encoder structures in the paper. So if you are free of GPU resource limitation, we recommend to use the large sliding window size for training.


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

- Photo-Realisitc Videos by [Video-to-Video](https://github.com/NVIDIA/vid2vid) </br>
We map the generated skeleton dances to the photo-realistic videos by [Video-to-Video](https://github.com/NVIDIA/vid2vid). Specifically, We record a random dance video of a team memebr to train the vid2vid model. Then we generate photo-realistic videos by feeding the generated skeleton dances to the trained vid2vid model. Note that, our team member has authorized us the usage of her portrait in following demos. 
<p align='center'>
  <img src='imgs/skeleton-1.gif' width='428'/>
  <img src='imgs/kazuna-crop-1.gif' width='400'/>
  <img src='imgs/skeleton-2.gif' width='429'/>
  <img src='imgs/kazuna-crop-2.gif' width='400'/>
</p>

- Driving 3D model by applying [3D human pose estimation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ci_Optimizing_Network_Structure_for_3D_Human_Pose_Estimation_ICCV_2019_paper.pdf)  and Unity animation to generated skeleton dances.


### Citation
If you find this work useful for your research, please cite the following paper:
```
@inproceedings{
  huang2021,
  title={ Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning},
  author={Ruozi Huang and Huang Hu and Wei Wu and Kei Sawada and Mi Zhang and Daxin Jiang},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
