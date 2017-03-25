from __future__ import print_function

import cPickle as pickle
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from random import shuffle

from PIL import Image

def load_database(data_dir = 'aligned_64'):

    persons = os.listdir(data_dir)
    persons.remove('.DS_Store')
    nb_pers = len(persons)
    print(nb_pers)
    DataBase = np.zeros(nb_pers).tolist()

    #DataBase[0] = np.zeros((100,3,96,96))
    #DataBase[1] = np.zeros((50, 3, 96, 96))
    #print(DataBase[1][0,0,:,:])
    #print(len(DataBase[1]))

    # Nombre min de photo pour une personne = 24

    i = 0
    for person in persons[0:3]:
        fullPersonPath = os.path.join(data_dir, person)
        pictures = os.listdir(fullPersonPath)
        if pictures.count('.DS_Store') > 0:
            pictures.remove('.DS_Store')
        print(person)
        nb_pics = len(pictures)
        print(nb_pics)

        person_pics = np.zeros((nb_pics,3,64,64))
        j = 0
        for pic in pictures:
            fullPicPath = os.path.join(fullPersonPath,pic)
            img = Image.open(fullPicPath)
            img = np.asarray(img, dtype='float32') / 256.
            person_pics[j, 0,: , :] = img[:, :, 0]
            person_pics[j, 1, :, :] = img[:, :, 1]
            person_pics[j, 2, :, :] = img[:, :, 2]
            j = j + 1
        DataBase[i] = person_pics
        i = i + 1

    return DataBase

DataBase = load_database()

# print(len(DataBase))
# print(DataBase[0].shape)
#
# print("saving DataBase to file ...")
# #with open("database.pck",'wb') as output:
#
# file = open("database.pck",'wb')
# pickle.dump(DataBase,file,pickle.HIGHEST_PROTOCOL)
# file.close()
#
# file2 = open("database.pck",'rb')
# print("loading DataBase from file ...")
# #with open("database.pck",'wb') as input1:
# DataBase_copy = pickle.load(file2)
# file2.close()
#
# print(len(DataBase_copy))
# print(DataBase_copy[0].shape)

# print("checking equality ...")
# for i in range(len(DataBase)):
#     if not np.all(DataBase[i] == DataBase_copy[i]):
#         print("NOOOOOO")

def select_random_person(nb_pers_sel, DataBase):

    # nb_pers_total = len(DataBase)
    nb_pers_total = 3

    indices = range(nb_pers_total)
    shuffle(indices)
    selected_indices = indices[0:nb_pers_sel]
    print(selected_indices)
    return list(DataBase[i] for i in selected_indices)

def select_random_pic_for_each_pers(nb_pic, DataBase):
    nb_pers = len(DataBase)

    batch = np.zeros((nb_pers*nb_pic,3,64,64))

    i = 0
    for person in DataBase:
        nb_pic_total_pers = len(person)
        indices = range(nb_pic_total_pers)
        shuffle(indices)
        selected_indices = indices[0:nb_pic]
        batch[i*nb_pic:(i+1)*nb_pic,:,:,:] = person[selected_indices]
        i = i + 1

    return batch

#print(DataBase[0][0,0,:,:])

selected_person = select_random_person(2,DataBase)

# print("selected_person :")
# print(len(selected_person))
# print(selected_person[0].shape)
# print(selected_person[1].shape)

batch = select_random_pic_for_each_pers(10, selected_person)

# print("batch :")
# print(batch.shape)

# print("save image ...")
# tmp = np.zeros((96,96,3))
# tmp[:,:,0] = batch[19][0,:,:]
# tmp[:,:,1] = batch[19][1,:,:]
# tmp[:,:,2] = batch[19][2,:,:]
# img = Image.fromarray((tmp * 255).astype(np.uint8))
# img.save('test.png')
# print("done !")

# img = Image.open("positive.png")
#
# print(img.size, img.format)
# img = np.asarray(img, dtype='float64') / 256.
# print(img.shape)
# #img = img.reshape((96,96,3))
# #img = img.reshape((3,96,96))
# print(img.shape)
# print(img.size)
#
#
#
# #new_img = np.zeros((1,96,96,3))
# new_img = np.zeros((1,3,96,96))
#
# print(new_img.shape)
#
# new_img[0,0,:,:] = img[:,:,0]
# new_img[0,1,:,:] = img[:,:,1]
# new_img[0,2,:,:] = img[:,:,2]


label = np.array([1,2,1,4,5,6,7,3,2,5,2,3,0,1,2,1,4,5,6,7],dtype='int32');


def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    print("input layer")
    l_in = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    print("first full layer")
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    print("second full layer")
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    print("output layer")
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

print("building model ...")

network = build_cnn(input_var)

print("prediction")
prediction = lasagne.layers.get_output(network)

#pred_norm, updates = theano.scan(lambda x_i: x_i/T.sqrt((x_i ** 2).sum()), sequences=[prediction])
#print(len(prediction))

#prediction[1] = prediction[1]/np.linalg.norm(prediction)

print(lasagne.layers.get_output_shape(network))
print("loss1")
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
print("loss2")
loss = loss.mean()

print("params")
params = lasagne.layers.get_all_params(network, trainable=True)
print("updates")
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

print("training on the image ...")
train_fn = theano.function([input_var, target_var], loss, updates=updates)

#img.reshape(1,96,96,3)
train_err = train_fn(batch,label)

print("  training loss:\t\t{:.6f}".format(train_err/1))
