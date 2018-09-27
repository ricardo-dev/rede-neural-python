#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:37:54 2018

@author: ricardo
"""

###
#  Entradas e saidas já determinadas pois é aprendizado supervisionado
#  Valores referente a porta lógica AND - vide caderno de estudo
###
import numpy as np

### Dados para operador AND
entradas        = np.array([[0,0],[0,1],[1,0],[1,1]])
saida           = np.array([  0,    0,    0,    1])

### Dados para operador OR
##entradas        = np.array([[0,0],[0,1],[1,0],[1,1]])
##saida           = np.array([   0,    1,    1,    1])

pesos           = np.array([0.0, 0.0])
taxaAprendizado = 0.1

def stepFunction(soma):
    '''
        funcao stepFunction - 'all-or-nothing'
        se retorno = 1, registro pertence a classe A
        se retorn  = 0, registro pertence a classe B
    '''
    if(soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)  ### Equivalente a funcao soma neural - produto escalar
    return stepFunction(s) 

def treinar():
    '''
        Funcao para fazer o ajuste de pesos
        do Perceptron, rede neural de uma camada
    '''
    erroTotal = 1
    while (erroTotal != 0): ## erro != 0 pois a base de dados pequena, mas 
        erroTotal = 0       ## geralmente utiliza-se aproximacao
        for i in range(len(saida)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saida[i] - saidaCalculada)
            erroTotal += erro
            if(erroTotal != 0):
                for j in range(len(pesos)):
                    pesos[j] = pesos[j] + (taxaAprendizado*entradas[i][j]*erro)
                    print('Peso atualizado: '+str(pesos[j]))
        print('Total de erros: '+str(erroTotal))

treinar()
                
print('Rede neural treinada!!')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))






