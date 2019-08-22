# adv_gan_keras

Keras implementation of AdvGAN by Xiao et al. (2018) https://arxiv.org/abs/1801.02610.
Note that the architecture of the generator produces a perturbation to then add to an original input image, instead of generating the perturbed image.

The generator works to minimize three losses: a hinge loss using the L2-norm between the original and its perturbed image, the GAN loss using the binary cross-entropy of the discriminator's real/fake classification, and the adversarial loss using the binary cross-entropy of a target's 1/3 classification; the generator forces the target to misclassify a 1 as a 3 and vice versa.

This implementation currently handles the binary-class case. The generator and discriminator of AdvGAN are implemented on a collection of 12,000 images of 1s and 3s from the MNIST dataset of handwritten digits.

![Pertubations on misclassified digits become imperceptible](https://raw.githubusercontent.com/niharikajainn/adv_gan_keras/master/45_epochs_perturbations.gif)

Sample of GAN-generated images from epoch 5 to 45. These are all misclassified by the target. The generated perturbations are detectable at first. With training, the model learns to create smoother, imperceptible perturbations to add to the original images.
