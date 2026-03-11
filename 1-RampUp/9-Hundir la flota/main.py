import time
import random
import os

import funciones
from constantes import LISTA_BARCOS, TAMANIO
import clases
from pyfiglet import figlet_format


os.system('cls')
tablero_jugador = clases.Tablero ("Naia",TAMANIO,LISTA_BARCOS)
tablero_PC = clases.Tablero ("PC",TAMANIO,LISTA_BARCOS)
tablero_jugador.inicializa_tableros()
tablero_PC.inicializa_tableros()
tablero_jugador.coloca_barcos_random()
# tablero_PC.coloca_barcos_manual() #para comprobar rápido que acaba la partida
tablero_PC.coloca_barcos_random()
print(figlet_format("HUNDIR LA FLOTA"))
print ("¡Bienvenido a hundir la flota!")
print ("El primer jugador que consiga hundir todos los barcos del oponente ganará el juego")
print ("Comienzas tú disparando")
input ("Pulsa cualquier tecla para comenzar a jugar: ")
turno = 1 # variable que controla el turno: 1 - turno del jugador, 2 - turno de la máquina
jugando = True
while (True):
    if turno == 1: 
        opcion=funciones.menu_opciones()
        if opcion == 1:
            os.system("cls")
            funciones.mensaje_turno_jugador ()
            tablero_jugador.imprime_tablero (0) #me imprime mis disparos para ver a donde tirar
            fila = funciones.pide_coordenada()
            columna = funciones.pide_coordenada ()
            acierto = tablero_PC.disparo (fila,columna) # Dispara. True si hay barco, False si no
            tablero_PC.actualiza_tablero (acierto,fila,columna) # Actualiza el tablero oponente
            tablero_jugador.registra_disparo (acierto, fila, columna) # Me guardo mi disparo
            #tablero_jugador.imprime_tablero (0) # muestro en pantalla el disparo
                    
            if (acierto):
                #Compruebo si quedan barcos
                if(tablero_PC.es_fin_partida ()):
                    os.system ('cls')
                    print(figlet_format("¡GANASTE LA PARTIDA!"))
                    break
            else:
            #si el jugador ha fallado el turno va a la máquina
                turno = 2
                time.sleep(3)
        if opcion ==0:
            print ("Juego finalizado")
            break
    if turno == 2:
        os.system('cls')
        funciones.mensaje_turno_PC ()
        fila,columna = funciones.disparo_aleatorio ()
        time.sleep (1)
        acierto = tablero_jugador.disparo (fila,columna)
        tablero_jugador.actualiza_tablero (acierto, fila, columna)
        tablero_jugador.imprime_tablero (1)
        if (acierto):
            time.sleep (2) # que de tiempo a leer que ha acertado
            #Compruebo si quedan barcos
            if(tablero_jugador.es_fin_partida ()):
                print ("La máquina te ha ganado :-(")
                break
        else:
        #si la máquina falla el turno vuelve al jugador
            turno = 1 
            input ("Te toca. Pulsa cualquier tecla... ")