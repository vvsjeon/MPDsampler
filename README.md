# [ICLR 2026] Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening

[Paper](https://openreview.net/pdf?id=GRElsj9W2t) | [Project page](https://vvsjeon.github.io/MPD/)

Official PyTorch implementation of "Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening".

<p align="center" width="100%">
    <video src='./assets/teaser.mp4' width='99%' controls autoplay loop muted></video>
</p>


### 1. Environment Setup
Our source code is based on [generative-models](https://github.com/Stability-AI/generative-models).  
Please clone the repository and place `mpd_par.py` and `mpd_seq.py` into the directory `scripts/sampling`.  
Follow the environment setup instructions provided in the [generative-models](https://github.com/Stability-AI/generative-models).

---

### 2. Pre-trained Model
Download the Stable Video Diffusion (SVD-XT) weights from [here](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt).  
Update the path to the downloaded model in the `ckpt_path` field of  
`scripts/sampling/configs/svd_xt.yaml`.

---

### 3. Inference
To run inference:

```
python scripts/sampling/mpd_par.py     ### Based on TRF
```
or
```
python scripts/sampling/mpd_seq.py     ### Based on ViBiD
```
+ The paths to the source frames should be specified using the flags ```input_start_path``` and ```input_end_path```. You can find some example pairs of start and end frames in the `examples/` folder.
+ You can adjust the ```fps_id``` (approximately between 6 and 24) according to the specific use case.


## Citation

```
@inproceedings{jeon2026motion,
    title={Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening},
    author={Wooseok Jeon and Seunghyun Shin and Dongmin Shin and Hae-Gon Jeon},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=GRElsj9W2t}
}
```
