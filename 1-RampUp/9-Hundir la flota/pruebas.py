import time
import random
import os

import funciones
from constantes import LISTA_BARCOS, TAMANIO
import clases
# Coloca los barcos random
tabla = clases.Tablero ("Naia", TAMANIO, LISTA_BARCOS)
tabla.inicializa_tableros()
tabla.coloca_barcos_random()
tabla.imprime_tablero(1)

