# Test-time model adaption for image reconstruction using self-supervised adaptive layers

[Test-time model adaption for image reconstruction using self-supervised adaptive layers](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05391.pdf)

Yutian Zhao,Tianjing Zhang and Hui Ji, National University of Singapore
(ECCV 2024)

## Content
* [Background](#Background)
* [Dataset](#Dataset)
* [Requirements](#Requirements)
* [Reference](#Reference)

## Overview
Image reconstruction from incomplete measurements is a basic task in medical imaging. While supervised deep learning proves to be a powerful tool for image reconstruction, it demands a substantial number of latent images for training. To extend the application of deep learning to medical imaging where collecting latent images poses challenges, this paper introduces an self-supervised test-time adaptation approach. The proposed approach leverages a pre-trained model on an external dataset and efficiently adapts it to each test sample for optimal generalization performance. Model adaption for an unrolling network is done with additional lightweight adaptive linear layers, enabling efficient alignment of testing samples with the distribution targeted in the pre-trained model. This approach is inspired by the connection between linear convolutional layer and Wiener filtering. Extensive experiments showed significant performance gain of the proposed method over other unsupervised methods and model adaptation techniques in two medical imaging tasks.
![image](https://github.com/XinranQin/DualDomainSS/blob/main/images/CS.png)

## Dataset
Dataset: [Training dataset](https://drive.google.com/drive/folders/1duLbfUDoUyN8JaCpVq1JwS_qIzKWB30M?usp=sharing "悬停显示")  
 
## Requirements
RTX3090 Python==3.8.0 Pytorch==1.8.0+cu101 scipy==1.7.3  


## References

```
@inproceedings{10.1007/978-3-031-72913-3_7,
author = {Zhao, Yutian and Zhang, Tianjing and Ji, Hui},
title = {Test-Time Model Adaptation for Image Reconstruction Using Self-supervised Adaptive Layers},
year = {2024},
isbn = {978-3-031-72912-6},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-72913-3_7},
doi = {10.1007/978-3-031-72913-3_7},
}
```

