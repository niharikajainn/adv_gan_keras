# adv_gan_keras

Working Keras replication of AdvGAN by Xiao et al. (2018) https://arxiv.org/abs/1801.02610.

The generator and discriminator of AdvGAN is currently implemented on 1s and 3s from the MNIST dataset of handwritten digits. The target classifier (and adversarial loss) is not yet implemented, so this can currently be treated as an ordinary GAN.

![From random noise to digits](https://raw.githubusercontent.com/niharikajainn/adv_gan_keras/master/35_epochs_training.gif)

Sample of GAN-generated images from epoch 0 to 35. These start looking close to random noise and become sharper, well-formed digits through more training.
