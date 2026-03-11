import random
from constantes import TAMANIO

def pide_coordenada ():
    '''
        Pide coordenada al jugador y la devuelve 
        Comprueba que sea válida y esté en el rango de 0 a 9. Hasta que no sea así no
        finaliza la función
    '''
    while True:
        try:
            coord = int (input("Introduce la coordenada a la que quieres disparar. De 0 a 9: "))
            if 0<=coord<=9:
                return coord
            else:
                print("La coordenada tiene que estar entre 0 y 9. Introdúcela de nuevo: ")
        except:
            print("La coordenada tiene que estar entre 0 y 9. Introdúcela de nuevo: ")

def disparo_aleatorio ():
    i = random.randint (0,TAMANIO -1)
    j = random.randint (0,TAMANIO-1)
    return i,j

def menu_opciones ():
    print ("Es tu turno. ¿Qué quieres hacer?:")
    print ("1 - Jugar")
    print ("0 - Salir")
    while (True):
        try:
            opcion = int(input ("Introduce opción: "))
            if opcion== 0 or opcion ==1:
                return opcion
            else:
                print ("Introduce una opción válida: ")
        except:
            print ("Introduce una opción válida: ")

def mensaje_turno_jugador ():
            print ("-"*42)
            print ()
            print ("              TE TOCA DISPARAR")
            print ()
            print ("-"*42)
            print ()
            print ()

def mensaje_turno_PC ():
            print ("-"*42)
            print ()
            print ("             LA MÁQUINA ESTÁ DISPARANDO")
            print ()
            print ("-"*42)
