import random
from constantes import TAMANIO
import os

class Tablero:
    def __init__ (self, id:str, tamanio:int, barcos:dict):
        self.id = id
        self.size = tamanio
        self.mis_barcos = barcos
        self.tabla_barcos = []
        self.tabla_disparos = []
    
    def inicializa_tableros (self):
        #Creamos los dos vacíos
        for i in range (self.size):
            fila = []
            fila2 = []
            for j in range (self.size):
                fila.append(" ")
                fila2.append ("_")
            self.tabla_barcos.append (fila)
            self.tabla_disparos.append (fila2)
    
    def coloca_barcos_manual (self):
        # Función de prueba para comprobar que se acaba el juego rápido
        self.tabla_barcos[1][0] = "B"
        self.tabla_barcos[1][1] = "B"
        self.tabla_barcos [9][9] = "B"     
            
    def coloca_barcos_random (self):
        # A partir de una posición inicial y una orientación calcula las posiciones donde iría el barco
        # Las vamos guardando en una lista
        # Se comprueba si son válidas
        # Si es que sí, se colocan
        orientaciones = ["N","S","E","O"]
           
        for n in self.mis_barcos.values():
            colocando = True # controla si el barco se ha podido colocar
            while (colocando):
                i = random.randint (0,TAMANIO-1)
                j = random.randint (0,TAMANIO-1)
                dir = random.choice (orientaciones)
                #Variable para guardar las posiciones
                barco = []
                barco.append ([i,j]) #guardo la posicion inicial
                if n != 1: #si el barco tiene más de una posición calculo el resto
                    for posicion in range(n-1): #ya he guardado la primera posicion, por eso quito uno. 
                        match dir:
                            case "N":
                                fila = i+1
                                col = j
                            case "S":
                                fila = i-1
                                col = j
                            case "E":
                                fila = i
                                col = j+1
                            case "O":
                                fila =i
                                col= j-1
                        barco.append ([fila,col])
                        i=fila
                        j=col
                # Compruebo que todas las posiciones están en la lista
                # que en ninguna posición de la lista hay ya un barco
                # que no hay barcos en las celdas contiguas
                valido = True
                for i,j in barco:
                    if (
                        (0> i) or (i> TAMANIO-1) or 
                        (0> j) or (j > TAMANIO-1) 
                        or self.tabla_barcos [i][j] == "B" 
                        or not self.comprueba_celdas_vecinas (i,j)
                    ):
                        valido = False
                        break 
                #coloca el barco
                if (valido):
                    for i, j in barco:
                        self.tabla_barcos [i][j] = "B"
                    colocando = False #barco colocado   
    
    def comprueba_celdas_vecinas (self,i,j): 
        # Comprueba las 8 celdas vecinas al disparo i,j: [i-1][j-1],[i-1][j],[i-1][j+1]...etc
        # Si la celda está fuera del tablero o está vacía devuelve es ok
        # Devuelve true si podríamos colocar un barco y False si no
        if (
            ((i-1 < 0 or j-1 < 0) or self.tabla_barcos[i-1][j-1] != "B") and 
            ((i-1 < 0) or self.tabla_barcos [i-1][j] != "B") and 
            ((i-1 < 0 or j+1 > 9) or self.tabla_barcos [i-1][j+1] != "B") and 
            ((j-1 < 0) or self.tabla_barcos [i][j-1] != "B") and 
            ((j+1 > 9) or self.tabla_barcos[i][j+1] !="B") and 
            ((i+1 > 9 or j-1 < 0) or self.tabla_barcos [i+1][j-1] != "B") and 
            ((i+1 > 9) or self.tabla_barcos [i+1][j] != "B") and 
            ((i+1 > 9 or j+1 > 9) or self.tabla_barcos[i+1][j+1] != "B")

            ):
                return True
        else:
            return False   
    def imprime_tablero (self,selecc):
        #Uso el mismo método para imprimir los dos tableros dependiendo del parámetro
        # 1 me imprime el estado de mis barcos
        # 0 (otro) me imprime donde he disparado
        if selecc == 1:
            tablero = self.tabla_barcos
        else:
            tablero = self.tabla_disparos

        #os.system('cls')

        # Printa tb una fila de 0 al 9 arriba y el 0 a 9 al principio de la columna para facilitar a donde
        # disparar
        print ("   0   1   2   3   4   5   6   7   8   9 ")
        for i in range (len(tablero)):
            print (i, end=" ")
            for j in range (len (tablero[i])):
                print (f"[{tablero[i][j]}]", end= " ") 
            print() 


    def disparo (self, i,j):
        '''Devuelve true si el disparo ha tocado barco o False si ha tocado agua'''
        if self.tabla_barcos [i][j] == "B":
            return True
        else: # puede haber " " o "O" si ya había disparado ahí
            return False
    
    def actualiza_tablero (self, disparo, i, j): 
        '''Qué barcos me han tocado'''
        if (disparo):
            self.tabla_barcos [i][j] = "X"
            print (f"Tocado en la posición fila: {i}, columna {j}")
        else:
            self.tabla_barcos [i][j] = "O"
            print ("Agua")

    def registra_disparo (self, disparo, i, j):
        ''' Para ver dónde voy disparando'''
        if (disparo):
            self.tabla_disparos [i][j] = "X"
        else:
            self.tabla_disparos [i][j] = "O"
    
    def es_fin_partida (self):
        for fila in self.tabla_barcos:
            if "B" in fila:
                return False
        return True 