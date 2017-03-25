def choose_triplet(batch_features,nb_pic_per_pers,nb_pers):

    size_batch = len(batch_features)

    triplets = np.zeros((nb_pic_per_pers*(nb_pic_per_pers-1)*nb_pers,3),dtype='int32')

    distances = np.zeros((size_batch,size_batch),dtype='float32')

    i = 0
    current_triplet_index = 0
    for features1 in batch_features:
        j = 0
        for features2 in batch_features:
            if i/nb_pic_per_pers == j/nb_pic_per_pers:
                if i != j:
                    triplets[current_triplet_index,0] = i
                    triplets[current_triplet_index,1] = j
                    dist = np.linalg.norm(features1 - features2)
                    current_triplet_index = current_triplet_index + 1
                    distances[i,j] = dist
                j = j + 1
            else:
                dist = np.linalg.norm(features1 - features2)
                distances[i,j] = dist
                distances[j,i] = dist
                j = j + 1
        i = i + 1

    for t in range(len(triplets)):
        anchor = triplets[t,0]
        positive = triplets[t,1]

        dist_anch_pos = distances[anchor,positive]
        best_semi_hard_dist = 10000
        best_semi_hard_pic = 0
        for neg in range(size_batch):
            if not neg/nb_pic_per_pers == anchor/nb_pic_per_pers:
                dist_anch_neg = distances[anchor, neg]
                if dist_anch_neg > dist_anch_pos:
                    if dist_anch_neg < best_semi_hard_dist:
                        best_semi_hard_pic = neg
                        best_semi_hard_dist = dist_anch_neg

        triplets[t,2] = best_semi_hard_pic

    return triplets