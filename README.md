# adv_gan_keras

Working Keras implementation of AdvGAN by Xiao et al. (2018) https://arxiv.org/abs/1801.02610.
The key difference between this implementation and the architecture from the paper is that the generator here does not produce a perturbation to then add to an original input image, but rather produces the perturbed image.

The generator works to minimize two losses: a hinge loss using the L2-norm between the original and its perturbed image and the GAN loss using the binary cross-entropy of the discriminator's real/fake classification. The third loss yet to be implemented is the adversarial loss using the binary cross-entropy of a target's 1/3 classification; the goal is the generator will force the target to misclassify a 1 as a 3 and vice versa.

The generator and discriminator of AdvGAN are currently implemented on a collection of 12,000 images of 1s and 3s from the MNIST dataset of handwritten digits.

![From random noise to digits](https://raw.githubusercontent.com/niharikajainn/adv_gan_keras/master/35_epochs_training.gif)

Sample of GAN-generated images from epoch 0 to 35. These start looking close to random noise and become sharper, well-formed digits through more training.
