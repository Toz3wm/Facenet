import face_test2 as t


def test3prec(nb_files):
    str1 = 'v3model'
    str2 = '.npz'
    print("Starting test differents modeles (t)")
    num_mod = 0
    res3pres = []

    for i in range(nb_files):
        res3pres.append(t.launchTests(str1+str(i*50)+str2, t.dbs, t.dbd, threshold = 1.19))
        print("model tested :"+str1+str(i*50)+str2)
    print('Liste des  3 precisions en faisant varier le modele:')
    print(res3pres)
    return(res3pres)


def testseuilVariable(mini,maxi,model, nb):
    print("Starting test seuilvariable")
    seuil = list(mini+i*(maxi-mini)/float(nb) for i in range(nb))
    res3presS = list()
    res3presD = list()
    res3presSD = list()
    for i in range(nb):
        z = t.launchTests(model, t.dbs, t.dbd, threshold = seuil[i])
        res3presS.append(z[0])
        res3presD.append(z[1])
        res3presSD.append(z[2])
        print("Seuil teste : "+str(seuil[i]))
    print("Liste des 3 precisions pour le seuil qui varie:")
    
    return((res3presS, res3presD, res3presSD))



#presOvertime = test3prec(18)
#presOverSeuil = testseuilVariable(0,2, 'v3model950.npz',20)

#presOverSeuilZoom = testseuilVariable(0.9,1.3, 'v3model950.npz',16)


presOverSeuilZoom = testseuilVariable(0.5,1.3, 'v3model950.npz',50)

