#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:44:20 2018

@author: ricardo
"""
import numpy as np

## indicado para perceptron
## P.L.S
## Transfer function
def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

## indicada para retornar probabilidades
## resultado final para problemas binarios (duas classes)
## P.N.L.S
def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

## resultado final para problemas binarios
## valores entre -1 e 1
## entradas negativas -> melhor mapeamento
def tahnFunction(soma):
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma) + np.exp(-soma))

## retorna 0 ou maior que zero
## nao existe valor maximo
## mais usada com R.N Convolucional CV e Deep Learning
def relufunction(soma):
    if (soma >= 0):
        return soma
    return 0

## retorna o valor passado, sem alteracao
## problemas de regressao
def linearFunction(soma):
    return soma

## Deep learning -> muito usada para problemas nao binaria
## baseia na probabilidade da classe
## Rede Neural Convulacional
def softMaxFunction(soma):
    esoma = np.exp(soma)
    return esoma / esoma.sum()

def teste(x):
    print(stepFunction(x))
    print(sigmoidFunction(x))
    print(tahnFunction(x))
    print(relufunction(x))
    print(linearFunction(x))
    
    valores = [7.0, 2.0, 1.3]
    print(softMaxFunction(valores))

teste(2.1)