from __future__ import division
#python 2.7.17rc1
import scipy.io as scipy
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import math
#import matplotlib.pyplot as plt


def app():
    mat = scipy.loadmat('data/grupoDados1.mat')
    dadosTest = mat['grupoTest']
    dadosTrain = mat['grupoTrain']
    rotuloTrain = mat['trainRots']
    rotuloTest = mat['testRots']
    
    #normalizando (0, 1)
    grupoTrainN = normaliza(dadosTrain)
    
    #k = 1
    print("K 1")
    classes = meuKnn(dadosTrain, rotuloTrain, dadosTest, 1)
    precisao = acurracia(classes, rotuloTest)
    print(precisao)
    print(classes)

    #k = 10
    print("K 10")
    classes = meuKnn(dadosTrain, rotuloTrain, dadosTest, 10)
    precisao = acurracia(classes, rotuloTest)
    print(precisao)
    print(classes)

def acurracia(rotulosPrevistos, rotulosTest):
    #calcula a % de acertos
    rl = len(rotulosPrevistos)
    acertos = 0
    for i in range(rl):
        if rotulosPrevistos[i] == rotulosTest[i]:
            acertos += 1
    return acertos / rl

def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots() 
    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red' , marker='^')
    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue' , marker='+')
    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.')
    plt.show()

def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []
    for idx in range(0, len(dados)):
        if(rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])        
    return ret

def normaliza(dados):
    p = MaxAbsScaler()
    p.fit(dados)
    return p.transform(dados)

def meuKnn(dadosTrain, rotuloTrain, dadosTest, k):
    dadosTrainL = len(dadosTrain)
    dadosTestL = len(dadosTest)
    distancias = [ [ 0 for i in range(dadosTrainL) ] for j in range(dadosTestL) ] 
    #distancias = []
    classes = []
    #calculando distancias
    for i in range(dadosTestL):
        #distancias.append({})
        for j in range(dadosTrainL):
            dist = dist_euclidiana(dadosTest[i], dadosTrain[j])
            #amarrando rotulo com distancia
            distancias[i][j] = [rotuloTrain[i][0], dist]
            idx_next = j - 1
            #ordenando
            while(True):
                if idx_next < 0:
                    break
                if distancias[i][idx_next][1] > dist:
                    aux = distancias[i][idx_next]
                    distancias[i][idx_next] = [rotuloTrain[j][0], dist]
                    distancias[i][idx_next + 1] = aux
                    idx_next -= 1
                    continue
                else:
                    break
        classe = [0,0,0]
        for g in range(k):
            idx = (distancias[i][g][0]) - 1
            #print("idx: ", idx)
            classe[idx] += 1
        print(classe)
        idx = sorted(classe)[2]
        idx = classe.index(idx) + 1
        classes.append(idx)
    #print(distancias[0])
    return classes
        


def dist_euclidiana(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    diff = v1 - v2
    quad_dist = np.dot(diff, diff)
    return math.sqrt(quad_dist)

app()