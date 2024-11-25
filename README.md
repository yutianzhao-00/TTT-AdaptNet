# Test-Time Model Adaptation for Image Reconstruction Using Self-supervised Adaptive Layers

Pytorch implementation of paper "Test-Time Model Adaptation for Image Reconstruction Using Self-supervised Adaptive Layers" (ECCV 2024)


## Abstract
Image reconstruction from incomplete measurements is a basic task in medical imaging. While supervised deep learning proves to be a
powerful tool for image reconstruction, it demands a substantial number
of latent images for training. To extend the application of deep learning
to medical imaging where collecting latent images poses challenges, this
paper introduces an self-supervised test-time adaptation approach. The
proposed approach leverages a pre-trained model on an external dataset
and efficiently adapts it to each test sample for optimal generalization
performance. Model adaption for an unrolling network is done with additional lightweight adaptive linear layers, enabling efficient alignment of
testing samples with the distribution targeted in the pre-trained model.
This approach is inspired by the connection between linear convolutional
layer and Wiener filtering. Extensive experiments showed significant performance gain of the proposed method over other unsupervised methods
and model adaptation techniques in two medical imaging tasks.

## Contents

In this repository, all experiments from the original paper can be reproduced.

