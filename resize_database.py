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

def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.
    If the directory already exists, don't do anything.
    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == exc.errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

#mkdirP('aligned_64')

data_dir = 'aligned'

persons = os.listdir(data_dir)
if persons.count('.DS_Store') > 0:
    persons.remove('.DS_Store')

for person in persons[381:]:
    mkdirP('aligned_64/'+person)
    fullPersonPath = os.path.join(data_dir, person)
    pictures = os.listdir(fullPersonPath)
    indices = range(len(pictures))
    shuffle(indices)
    selected_indices = indices[0:min(len(pictures),100)]
    for pic in list(pictures[i] for i in selected_indices):
        if pictures.count('aligned/'+pic+'/.DS_Store') > 0:
            pictures.remove('aligned/'+pic+'/.DS_Store')
        fullPicPath = os.path.join(fullPersonPath, pic)
        img = Image.open(fullPicPath)
        img_64 = img.resize((64,64),Image.ANTIALIAS)
        img_64.save('aligned_64/'+person+'/'+pic)