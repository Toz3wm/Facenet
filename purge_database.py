def purge_database(data_dir = 'aligned_64_train'):

    persons = os.listdir(data_dir)[0:]
    if persons.count('.DS_Store') > 0:
        persons.remove('.DS_Store')
  
    for person in persons:
        fullPersonPath = os.path.join(data_dir, person)
        pictures = os.listdir(fullPersonPath)
        if pictures.count('.DS_Store') > 0:
            pictures.remove('.DS_Store')
        for pic in pictures:
            fullPicPath = os.path.join(fullPersonPath,pic)
            if os.path.getsize(fullPicPath) == 0:
                os.remove(fullPicPath)
            

    
    
    
