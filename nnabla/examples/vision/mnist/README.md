# MNIST Examples

---

## Overview

The examples listed below demonstrate several deep learning algorithms on MNIST
dataset, which is one of the most popular image classification datasets in the
machine learning community. The MNIST dataset will be automatically downloaded
when running any of the examples.

---

## Classification task (`classification.py` and `classification_*.py`)

This example demonstrates the training of hand-written digits classification on
MNIST dataset. The convolutional neural network takes 28x28 pixel grayscale
images as input and outputs the predictions of 10-way classification.

When you run the example by

```
python classification.py

```

you will see the training progress in the console (decreasing training and
validation error). The model will eventually reach around 1% validation error
rate.

Training can be dramatically sped up by using a CUDA GPU when the
nnabla_ext-cuda extension is installed. Run the above command with `-c
cuda.cudnn` option to let NNabla use GPU acceleration.

```
python classification.py -c cuda.cudnn
```

After the learning completes successfully, the results will be saved in
"tmp.monitor/". In this folder you will find model files "\*.h5" and result
files "\*.txt".

The classification example provides two choices of neural network architectures
to train, LeNet and ResNet. You can select it with the `-n` option. For more
details see the source code and the help produced by running with the `-h`
option.

You can also try training of various types of binary neural network
(`classification_bnn.py`) and quantized classification models
(`classification_qnn.py`) on the MNIST dataset:

* Binarized neural network (BNN)

  `classification_bnn.py` provides an example of training a BNN on MNIST

  ```
  python classification_bnn.py [-c cuda.cudnn] [-h|--help]
  ```

  The quantization method is chosen by `--net`. By default, a BinaryConnect
  version of a LeNet-style network is trained.

* Quantized neural network

  `classification_qnn.py` provides an example of training a QNN on MNIST

  ```
  python classification_qnn.py [-c cuda.cudnn] [-h|--help]
  ```

  By default a multiplierless LeNet-style network with 4-bit power-of-two values
  is trained using Incremental Network Quantization (INQ).

## Deep Convolutional GAN (`dcgan.py`)

This example shows how to learn and generate images using Generative
Adversarial Networks demonstrated on MNIST dataset. The image generator neural
network takes 256 dimension random vectors as inputs, and outputs 28x28 pixel
generated images. The discriminator neural network takes 28x28 pixel images as
inputs, and outputs the score of whether the inputs are real or generated.

Reference: "Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks". https://arxiv.org/abs/1507.00677

```
python dcgan.py [-c cuda.cudnn] [-h|--help]
```

After the learning completes correctly, the results will be saved in
"tmp.monitor.dcgan". In this folder you will find the discriminator model files
"discriminator_\*.h5", the generator model files "generator_\*.h5", log files
"\*.txt" and some generated images in "Fake-images".

## Feature Embedding (`siamese.py`)

This example demonstrates training of a neural network model that embeds an
image into a low dimensional feature space. The Siamese Neural Network takes a
pair of 28x28 pixel images as input, and outputs the feature embedding vectors.

```
python siamese.py [-c cuda.cudnn] [-h|--help]
```

After the learning completes successfully, the results will be saved in
"tmp.monitor.siamese".  In this folder you will find model files
"params_\*.h5", log files "\*.txt" and a 2d t-SNE image file "embed.png".

## Semi-Supervised Learning of Classification (`vat.py`)

This is a semi-supervised learning example demonstrating "Virtual Adversarial
Training".

Reference: "Distributional Smoothing with Virtual Adversarial Training"
https://arxiv.org/abs/1507.00677

```
python vat.py [-c cuda.cudnn] [-h|--help]
```

After the learning completes successfully, the results will be saved in
"tmp.monitor.vat". In this folder you will find model files "params_\*.h5" and
log files "\*.txt".
