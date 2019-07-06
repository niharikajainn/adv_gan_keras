from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras import layers, Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape, Activation, Lambda, LeakyReLU
from keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam, SGD
from keras.metrics import binary_accuracy
from keras import backend as K
import os, cv2, re, random
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed

class DCGAN():

    def __init__(self):
        #input image dimensions
        self.img_width = 28
        self.img_height = 28
        self.input_shape = (self.img_width, self.img_height, 1) #1 channel for grayscale

        optimizer_g = Adam(0.0002)
        optimizer_d = SGD(0.01)

        inputs = Input(shape=self.input_shape)
        outputs = self.build_generator(inputs)
        self.G = Model(inputs, outputs)
        self.G.summary()

        outputs = self.build_discriminator(self.G(inputs))
        self.D = Model(inputs, outputs)
        self.D.compile(loss=keras.losses.binary_crossentropy, optimizer = optimizer_d, metrics=[self.custom_acc])
        self.D.summary()

        outputs = self.build_target(self.G(inputs))
        self.target = Model(inputs, outputs)
        self.target.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.target.summary()

        self.stacked = Model(inputs=inputs, outputs=[self.G(inputs), self.D(self.G(inputs)), self.target(self.G(inputs))])
        self.stacked.compile(loss=[self.generator_loss, keras.losses.binary_crossentropy, keras.losses.binary_crossentropy], optimizer = optimizer_g)
        self.stacked.summary()

    def generator_loss(self, y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)
        #||G(x) - x||_2 - c, where c is user-defined. Here it is set to 0.3

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_true), K.round(y_pred))

    #build the cnn
    def build_discriminator(self, inputs):

        D = Conv2D(32, 4, strides=(2,2))(inputs)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Conv2D(64, 4, strides=(2,2))(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Flatten()(D)
        D = Dense(64)(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)
        return D


    def build_generator(self, inputs):
        #c3s1-8
        G = Conv2D(8, 3, padding='same')(inputs)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #d16
        G = Conv2D(16, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #d32
        G = Conv2D(32, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        residual = G
        #four r32 blocks
        for _ in range(4):
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G

        #u16
        G = Conv2DTranspose(16, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #u8
        G = Conv2DTranspose(8, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #c3s1-3
        G = Conv2D(1, 3, padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)
        G = layers.add([G, inputs])

        return G

    def build_target(self, inputs):
        f = Conv2D(64, 5, padding='same', activation='relu')(inputs)
        f = Conv2D(64, 5, padding='same', activation='relu')(f)
        f = Dropout(0.25)(f)
        f = Flatten()(f)
        f = Dense(128, activation='relu')(f)
        f = Dropout(0.5)(f)
        f = Dense(2, activation='softmax')(f)
        return f



    def get_batches(self, start, end, x_train, y_train):
        x_batch = x_train[start:end]
        Gx_batch = self.G.predict_on_batch(x_batch)
        y_batch = y_train[start:end]
        return x_batch, Gx_batch, y_batch


    def train_D_on_batch(self, batches):
        x_batch, Gx_batch, _ = batches

        #for each batch:
            #predict noise on generator: G(z) = batch of fake images
            #train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
            #train real images on disciminator: D(x) = update D params per classification for real images

        #Update D params
        self.D.trainable = True
        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(len(x_batch), 1)) ) #real=1, positive label smoothing
        d_loss_fake = self.D.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)) ) #fake=0
        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

        return d_loss #(loss, accuracy) tuple


    def train_stacked_on_batch(self, batches):
        x_batch, _, y_batch = batches
        flipped_y_batch = 1.-y_batch

        #for each batch:
            #train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        #Update only G params
        self.D.trainable = False
        self.target.trainable = False
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(flipped_y_batch)] )
        #stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(y_batch)] )
        #input to full GAN is original image
        #output 1 label for generated image is original image
        #output 2 label for discriminator classification is real/fake; G wants D to mark these as real=1
        #output 3 label for target classification is 1/3; g wants to flip these so 1=1 and 3=0
        return stacked_loss #(total loss, hinge loss, gan loss, adv loss) tuple


    def train_GAN(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train *2./255 - 1).reshape((len(x_train), 28, 28, 1)) #pixel values in range [-1., 1.] for D
        binary_indices = np.where(y_train == 1)
        x_ones = x_train[binary_indices][:6000]
        y_ones = np.zeros((6000, 1))
        binary_indices = np.where(y_train == 3)
        x_threes = x_train[binary_indices][:6000]
        y_threes = np.ones((6000, 1))
        x_train = np.concatenate((x_ones, x_threes)) #(12000, 28, 28, 1)
        y_train = np.concatenate((y_ones, y_threes)) #1=0, 3=1
        zipped = list(zip(x_train,y_train))
        np.random.shuffle(zipped)
        x_train, y_train = zip(*zipped)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        self.target.fit(x_train, to_categorical(y_train), epochs=5) #pretrain target

        epochs = 50
        batch_size = 128
        num_batches = len(x_train)//batch_size
        if len(x_train) % batch_size != 0:
            num_batches += 1

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            batch_index = 0

            for batch in range(num_batches - 1):
                start = batch_size*batch_index
                end = batch_size*(batch_index+1)
                batches = self.get_batches(start, end, x_train, y_train)
                self.train_D_on_batch(batches)
                self.train_stacked_on_batch(batches)
                batch_index += 1


            start = batch_size*batch_index
            end = len(x_train)
            x_batch, Gx_batch, y_batch = self.get_batches(start, end, x_train, y_train)

            (d_loss, d_acc) = self.train_D_on_batch((x_batch, Gx_batch, y_batch))
            (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_stacked_on_batch((x_batch, Gx_batch, y_batch))

            target_acc = self.target.test_on_batch(Gx_batch, to_categorical(y_batch))[1]
            target_predictions = self.target.predict_on_batch(Gx_batch) #(96,2)

            misclassified = np.where(y_batch.reshape((len(x_train) % batch_size, )) != np.argmax(target_predictions, axis=1))[0]
            print(np.array(misclassified).shape)
            print(misclassified)

            print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: %f\tAccuracy:%.2f%%" %(d_loss, d_acc*100., gan_loss, hinge_loss, adv_loss, target_acc*100.))

            if epoch == 0:
                self.save_generated_images("orig", x_batch, 'images')
            if epoch % 5 == 0:
                self.save_generated_images(str(epoch), Gx_batch, 'images')
                self.save_generated_images(str(epoch), Gx_batch[misclassified], 'misclass')


    def save_generated_images(self, filename, batch, dir):
        batch = batch.reshape(batch.shape[0], self.img_width, self.img_height)
        rows, columns = 5, 5

        fig, axs = plt.subplots(rows, columns)
        cnt = 0
        for i in range(rows):
            for j in range(columns):
                axs[i,j].imshow((batch[cnt] + 1)/2., interpolation='nearest', cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("%s/%s.png" % (dir, filename))
        plt.close()




if __name__ == '__main__':
    seed(5)
    set_random_seed(1)
    dcgan = DCGAN()
    dcgan.train_GAN()
