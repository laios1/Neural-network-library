import numpy as np 
import json




###################################################Neural Network#################################################################
def print_graphe_graphviz(SSize,w,b,X,results) : 
    to_print = "digraph G {\nnode [shape=record,width=.1,height=.1];\n"
    k = 0
    to_print = to_print + "subgraph cluster0 {\n"
    for i in range(len(X)):
        to_print = to_print + """node%d [label = "{<n>%d|%.2f|a<p>}"];\n""" % (k,i,X[i])
        k += 1
    to_print = to_print + "}\n"

    for i in range(len(SSize)-1) :
        to_print = to_print + "subgraph cluster%d {\n" % (i+1)
        for j in range(SSize[i+1]):
            to_print = to_print + """node%d [label = "{<n>%d/%d|%.2f|%.2f|a<p>}"];\n""" % (k,i,j,b[i][j],results[i][j])
            k += 1
        to_print = to_print + "}\n"
    
    totSSize = 0
    for i in range(len(SSize)-1) :
        for j in range(SSize[i]):
            for l in range(SSize[i+1]):
                to_print = to_print + """node%d -> node%d [label = %.2f];\n""" % (j+totSSize,l+totSSize+SSize[i],w[i][j][l])
        totSSize += SSize[i]
    to_print = to_print+ "}"
    return to_print

class neural_network(object):
    """
    ### crée un réseau de neurone : 

    `SSize` = [#input_layer, #hidden_layer_1,..., #hidden_layer_n, #output_layer]

    /!\ peut ne pas avoir d'hidden layer

    `LR`(Learning rate) ; LR = 1 par default 

    `act` (activation function) ; act = "sigmoid" par defaut ; (pour l'instant seulement sigmoid and ReLU)
    """
    def __init__(self,SSize,LR = 1,act = "sigmoid"):
        self.Size = SSize
        self.ActivationType = act

        self.w = [] #les poids/ taille : (nombre_de_couches-1)*(taille de la couche i)*(taille de la couche i+1)
        self.bias = [] #les correctifs d'érreurs / taille : (nb_couches-1)*(taille de la couche i)
        for i in range(len(self.Size)-1) : 
            self.w.append(np.random.randn(self.Size[i],self.Size[i+1])) #on met les poids en random 
            self.bias.append(np.random.randn(self.Size[i+1])) #on assigne les correctifs de chaques neurones 
        #self.bias.append(np.random.randn(self.Size[-1])) #on assigne les correctifs de la derniere couche 
        
        
        self.learningRate = LR 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []
        self.dError  = []


    #-------------------------------fonctions d'activations--------------------------------------

    def identite(self,s):
        return np.array(s)

    def identiteDerive(self,s):
        return np.array([1 for i in s])

    def ReLU(self,s):
        return np.array([i if i >= 0 else 0.1*i for i in s ])

    def ReLUDerive(self,s):
        return np.array([1 if i >= 0 else 0.1 for i in s ])

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidDerive(self,s):
        return self.sigmoid(s)*(1-self.sigmoid(s))

    def tanh(self,s):
        return (2/(1+np.exp(-2*s)))-1

    def tanhDerive(self,s):
        return 1-(self.tanh(s)*self.tanh(s))

    def activation(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoid(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLU(s)
        elif self.ActivationType == "id" :
            return self.identite(s)
        elif self.ActivationType == "tanh" :
            return self.tanh(s)

    def activationDerive(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoidDerive(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLUDerive(s)
        elif self.ActivationType == "id" :
            return self.identiteDerive(s)
        elif self.ActivationType == "tanh" :
            return self.tanhDerive(s)
        
    
    #-------------------evaluation de la fonction reseau de neurones en X--------------------------------
    def forward(self,X):
        #self.preAct = 0 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []

        self.preAct.append(np.dot(X,self.w[0]) + self.bias[0]) #produit matricielle 
        self.results.append(self.activation(self.preAct[0])) #activation du neurone (lissage des données)
        
        for i in range(1,len(self.Size)-2) : #on itere pour chaque neurone 
            self.preAct.append(np.dot(self.results[i-1],self.w[i]) + self.bias[i])
            self.results.append(self.activation(self.preAct[i]))
        
        self.preAct.append(np.dot(self.results[len(self.Size)-3],self.w[len(self.Size)-2]) + self.bias[len(self.Size)-2])
        self.results.append(self.preAct[len(self.Size)-2])

        return self.results[-1]


    #---------------------------Entrainement--------------------------------------

    def backward(self,X,y,o):
        """
        X : valeur d'entree 

        y : valeur attendu par le modele 

        o : valeur retourne par le model pour une entree donne
        """
        self.dError = [] #calcul de l'erreur 


        list.insert(self.dError,0, 2*(o-y))
        for i in range(1,len(self.Size)-1) : 
            list.insert(self.dError,0, np.dot(self.dError[0],self.w[-i].T) * self.activationDerive(self.preAct[-(i+1)]))

        self.w[0] -= np.dot(np.array(X,ndmin=2).T, np.array(self.dError[0],ndmin=2))*self.learningRate
        self.bias[0] -= self.dError[0]*self.learningRate
        

        for i in range(1,len(self.Size)-1) : 
            self.w[i] -= np.dot(np.array(self.results[i-1],ndmin=2).T, np.array(self.dError[i],ndmin=2))*self.learningRate
            self.bias[i] -= self.dError[i]*self.learningRate
        

    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)
        return np.abs(o-y)


    def error(self,X,y):
        o = self.forward(X)
        return np.abs(o-y)


    def predict(self,xIncunue):
        print(xIncunue)
        print(tuple(self.forward(xIncunue)))



    #--------------------------sauvegarder, charger et afficher le reseau de Neurones-----------------------------
    def print_NN(self,X):
        o = self.forward(X)
        print(print_graphe_graphviz(self.Size,self.w,self.bias,X,self.results))

    def save_NN(self, name = "Neural_Network_save"):
        with open(name+'.json', 'w', encoding='utf-8') as f:
            
            w2 = [self.w[i].tolist() for i in range(len(self.w))]
            b2 = [self.bias[i].tolist() for i in range(len(self.bias))]
            activation_f = self.ActivationType
            dic = {"weight" : w2, "bias" : b2,"act_f" : activation_f}
            json.dump(dic, f, ensure_ascii=False, indent=4)  

    def load_NN(self,name):
        with open(name+'.json') as f:
            data_loaded = json.load(f)
            self.w = [np.array(data_loaded["weight"][i]) for i in range(len(data_loaded["weight"]))]
            self.bias = [np.array(data_loaded["bias"][i]) for i in range(len(data_loaded["bias"]))]
            self.ActivationType = data_loaded["act_f"]

    

def print_shape(array):
    for i in range(len(array)):
        print(array[i].shape)
####################################################Deep-Qlearning############################################################  