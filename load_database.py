import sys
import os
import time

import numpy as np

from PIL import Image


# return the database as a list of size 'number of different people in the dataset'
# each element of the list is a numpy array [nb_pic_for_this_person, 3, 96, 96]
# (3 channels images of size 96*96)
# to get the fisrt pic of the first person do : DataBase[0][0,:,:,:]

def load_database(data_dir = 'aligned'):

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

        person_pics = np.zeros((nb_pics,3,96,96))
        j = 0
        for pic in pictures:
            fullPicPath = os.path.join(fullPersonPath,pic)
            img = Image.open(fullPicPath)
            img = np.asarray(img, dtype='float64') / 256.
            person_pics[j, 0,: , :] = img[:, :, 0]
            person_pics[j, 1, :, :] = img[:, :, 1]
            person_pics[j, 2, :, :] = img[:, :, 2]
            j = j + 1
        DataBase[i] = person_pics
        i = i + 1

    return DataBase