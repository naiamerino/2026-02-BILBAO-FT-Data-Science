import time
import random
import os

import funciones
from constantes import LISTA_BARCOS, TAMANIO
import clases

os.system('cls')
tablero_jugador = clases.Tablero ("Naia",TAMANIO,LISTA_BARCOS)
tablero_PC = clases.Tablero ("PC",TAMANIO,LISTA_BARCOS)
tablero_jugador.inicializa_tableros()
tablero_PC.inicializa_tableros()
tablero_jugador.coloca_barcos_manual()
tablero_PC.coloca_barcos_manual()
print ("¡Bienvenido a hundir la flota!")
print ("El primer jugador que consiga hundir todos los barcos del oponente ganará el juego")
print ("Comienzas tú disparando")
input = ("Pulsa cualquier tecla para comenzar a jugar: ")
turno = 1 # variable que controla el turno: 1 - turno del jugador, 2 - turno de la máquina
jugando = True
while (True):
    if turno == 1: # Turno jugador
        opcion=funciones.menu_opciones()
        if opcion == 1:
            tablero_jugador.imprime_tablero (0) #me imprime mis disparos para ver a donde tirar
            fila = funciones.pide_coordenada()
            columna = funciones.pide_coordenada ()
            acierto = tablero_PC.disparo (fila,columna) # Dispara. True si hay barco, False si no
            tablero_PC.actualiza_tablero (acierto,fila,columna)
            tablero_jugador.registra_disparo (acierto, fila, columna)
                    
            if (acierto):
                #Compruebo si quedan barcos
                if(tablero_PC.es_fin_partida ()):
                    print ("¡Has ganado la partida!")
                    break
            else:
            #si el jugador ha fallado el turno va a la máquina
                turno = 2
        if opcion ==0:
            print ("Juego finalizado")
            break
    if turno == 2:
        print ("La máquina está disparando")
        fila,columna = funciones.disparo_aleatorio ()
        time.sleep (2)
        acierto = tablero_jugador.disparo (fila,columna)
        tablero_jugador.actualiza_tablero (acierto, fila, columna)
        tablero_jugador.imprime_tablero (1)
        if (acierto):
            #Compruebo si quedan barcos
            if(tablero_jugador.es_fin_partida ()):
                print ("La máquina te ha ganado :-(")
                break
        else:
        #si la máquina falla el turno vuelve al jugador
            turno = 1 