#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 19:25:20 2018

@author: ricardo
"""

import numpy as np

def sigmoid(soma):
    '''
        funcao sigmoid rede neural
        retorna valores entre 0 e 1 (0.0, 0.1, ... , 0.8, 0.9, 1.0)
    '''
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    '''
        sig - resultado da funcao sigmoid
        retorna a derivada parcial
    '''
    return sig * (1 - sig)


### Entradas referente a porta XOR
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

### Saidas referente a porta XOR
saidas   = np.array([[0],[1],[1],[0]])

### Pesos da entrada para camada oculta [ [x1], [x2]]
#pesos0 = np.array([[-0.424, -0.740, -0.961] , [0.358, -0.577, -0.469]])

### Pesos da camada oculta para camada final
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])

### Pesos aleatorios (2 neuronios entrada, 3 neuronios ocultas)
pesos0 = 2*np.random.random((2,3)) - 1

pesos1 = 2*np.random.random((3,1)) - 1

### Utilizacao de iteracao para atualizar os pesos
### quando rede neural complexa, geralmente o erroTotal pode nunca ser 0 (perceptron)
### valor muito alto
### Alterar a epoca até mediaAbsoluta ser baixa o suficiente
epocas = 1000000
taxaAprendizagem = 0.6
momento = 1

### Training time
for i in range(epocas):
    ### auxiliar -> feedfoward
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0) # ja percorre sozinho
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    ### Calculo do erro -> minimizar media absoluta do erro
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(abs(erroCamadaSaida))
    #print("Erro: "+str(mediaAbsoluta))
    
    ### Derivada e Gradiente (Delta Cadamada de saida)
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    ### Derivada e Gradiente (Delta camada oculta)
    pesos1Transposta = pesos1.T # Transposta para ajustar dimensão
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    ### pesos atualizados com BATH Gradient descent, calculo do erro para todos os registros
    
    ### Backpropragation - camadaOculta para saida
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1*momento) + (pesosNovo1*taxaAprendizagem)
    
    ### Backpropagation - camadaEntrada para camadaOculta
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0*momento) + (pesosNovo0*taxaAprendizagem)

print("Taxa de acertos:",(1-mediaAbsoluta)*100,"%")
print("pesos camada entrada para camada oculta: "+str(pesos0))
print("pesos camada oculta para camada saida: "+str(pesos1))
print("Saida calculada: \n")
for i in camadaSaida:
    for j in i:
        print(round(j))
#print(round(camadaSaida, 5))

print('Rede neural treinada')
Entrada = np.array([[0,0], [0,1]])

CamadaEntrada = Entrada
SomaSinapse0 = np.dot(CamadaEntrada, pesos0)
CamadaOculta = sigmoid(SomaSinapse0)

SomaSinapse1 = np.dot(CamadaOculta, pesos1)
CamadaSaida = sigmoid(SomaSinapse1)

print('Resultado: \n')
print(CamadaSaida)


    
    
    
    
    
    
    