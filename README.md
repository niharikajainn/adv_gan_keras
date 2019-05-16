# adv_gan_keras

Working Keras replication of AdvGAN by Xiao et al. (2018) https://arxiv.org/abs/1801.02610.

The generator and discriminator of AdvGAN is currently implemented on 1s and 3s from the MNIST dataset of handwritten digits. The target classifier (and adversarial loss) is not yet implemented, so this can currently be treated as an ordinary GAN.

![Random noise at epoch 0](https://raw.githubusercontent.com/niharikajainn/adv_gan_keras/master/images/0.png)

Sample of GAN-generated images at epoch 0. These look close to random noise.


![Well-formed digits at epoch 35](https://raw.githubusercontent.com/niharikajainn/adv_gan_keras/master/images/35.png)

Sample of GAN-generated images at epoch 35. These look close to well-formed digits.
