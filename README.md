# [ICLR 2026] Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening

[Paper](https://openreview.net/pdf?id=GRElsj9W2t) | [Project page](https://vvsjeon.github.io/MPD/)

Official PyTorch implementation of "Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening".

<p align="center" width="100%">
    <video src='./assets/teaser.mp4' width='99%' controls autoplay loop muted></video>
</p>


### 1. Environment setup
Our code is built on [generative-models](https://github.com/Stability-AI/generative-models).  
Clone that repository and place `mpd_par.py` and `mpd_seq.py` into the `scripts/sampling` directory.  
Then follow the environment setup instructions provided in [generative-models](https://github.com/Stability-AI/generative-models).

---

### 2. Pre-trained model
Download the Stable Video Diffusion (SVD-XT) weights from [here](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt).  
Then set the `ckpt_path` field in `scripts/sampling/configs/svd_xt.yaml` to the path of the downloaded model.

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
+ Specify the source frames with the ```input_start_path``` and ```input_end_path``` flags. Example start–end frame pairs are provided in the `examples/` folder.
+ Adjust ```fps_id``` (roughly between 6 and 24) depending on your use case.


## Acknowledgements

The overall repository structure largely follows [ViBiDSampler](https://github.com/vibidsampler/vibid). We thank the authors for releasing their code.

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
