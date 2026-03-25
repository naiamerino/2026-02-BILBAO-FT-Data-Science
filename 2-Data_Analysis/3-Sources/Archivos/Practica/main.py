import os
import shutil
import funciones
import variables

funciones.crea_carpetas (variables.path)
archivos=funciones.lista_archivos (variables.path)
funciones.clasifica_archivos (archivos, variables.path)