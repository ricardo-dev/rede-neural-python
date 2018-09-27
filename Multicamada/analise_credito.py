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

def relu(soma):
    '''
        funcao RELU rede neural
        retorna valores entre 0 e soma - linear
    '''
    if (soma.any() < 0):
        return 0
    return soma

def reluDerivada(re):
    '''
        re - resultado da funcao relu
        retorna a derivada parcial
    '''
    if ( re.any() < 0 ):
        return 0
    return 1

### Entradas referente ao teste de analise de credito
entradas = np.array([ [3,1,1,1],
                      [2,1,1,2],
                      [2,2,1,2],
                      [2,2,1,3],
                      [2,2,2,3],
                      [3,2,1,1],
                      [1,2,1,3],
                      [1,1,2,3],
                      [1,1,1,1],
                      [1,1,1,2],
                      [1,1,1,3],
                      [3,1,1,2] ])

### Saidas referente ao teste de analise de credito
saidas   = np.array([ [1,0,0],
                      [1,0,0],
                      [0,1,0],
                      [1,0,0],
                      [0,0,1],
                      [1,0,0],
                      [0,0,1],
                      [0,0,1],
                      [1,0,0],
                      [0,1,0],
                      [0,0,1],
                      [1,0,0]  ])

### Pesos da entrada para camada oculta [ [x1], [x2]]
#pesos0 = np.array([[-0.424, -0.740, -0.961] , [0.358, -0.577, -0.469]])

### Pesos da camada oculta para camada final
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])

### Pesos aleatorios (2 neuronios entrada, 3 neuronios ocultas)
pesos0 = 2*np.random.random((4,4)) - 1

pesos1 = 2*np.random.random((4,3)) - 1

pesos2 = 2*np.random.random((4,3)) - 1 #***

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
    #camadaOculta = relu(somaSinapse0)
    
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
    
    ### Backpropragation - camadaOculta para saida
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1*momento) + (pesosNovo1*taxaAprendizagem)
    
    ### Backpropagation - camadaEntrada para camadaOculta
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0*momento) + (pesosNovo0*taxaAprendizagem)

print("Taxa de acertos:",(1-mediaAbsoluta)*100,"%")
#print("pesos camada entrada para camada oculta: "+str(pesos0))
#print("pesos camada oculta para camada saida: "+str(pesos1))
#print("Saida calculada: \n")
#for i in camadaSaida:
#    for j in i:
#        print(round(j))
#    print('\n')

print('Rede neural treinada')
#Entrada = np.array([[2,2,1,1], [1,2,2,2], [1,1,2,2], [3,1,1,1], [2,2,1,2]])

#CamadaEntrada = Entrada
#SomaSinapse0 = np.dot(CamadaEntrada, pesos0)
#CamadaOculta = sigmoid(SomaSinapse0)

#SomaSinapse1 = np.dot(CamadaOculta, pesos1)
#CamadaSaida = sigmoid(SomaSinapse1)

#print('Resultado: \n')
#print(CamadaSaida)
#for i in CamadaSaida:
#    for j in i:
#        print(round(j))
#    print('*')

def entrarValores():
    while(1): 
        v1 = int(input("Historico de credito:\n1 - Bom\n2 - Desconhecido\n3 - Ruim\n"))
        v2 = int(input("Divida:\n1 - Alta\n2 - Baixa\n"))
        v3 = int(input("Garantia:\n1 - Nenhum\n2 - Adequada\n"))
        v4 = int(input("Renda anual:\n1 - < 15.000\n2 - >=15.000 e <35.000\n3 - > 35.000\n"))
        Entrada = np.array([[v1,v2,v3,v4]])
        
        CamadaEntrada = Entrada 
        SomaSinapse0 = np.dot(CamadaEntrada, pesos0)
        CamadaOculta = sigmoid(SomaSinapse0)

        SomaSinapse1 = np.dot(CamadaOculta, pesos1)
        CamadaSaida = sigmoid(SomaSinapse1)
        
        if(round(CamadaSaida[0][0]) == 1):
            print("Risco de crédito - Alto")
        elif (round(CamadaSaida[0][1]) == 1):
            print("Risco de crédito - Moderado")
        else:
            print("Risco de crédito - Baixo")

entrarValores()