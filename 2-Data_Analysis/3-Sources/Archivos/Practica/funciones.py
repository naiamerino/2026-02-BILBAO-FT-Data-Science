import os
import shutil

def lista_archivos (path):
    lista_archivos = os.listdir (path)
    return (lista_archivos)

def crea_carpetas (path):
    os.mkdir(path+"\Documentos")
    os.mkdir(path+"\Imagenes")
    os.mkdir(path+"\SW")
    os.mkdir(path+"\Otros")


def clasifica_archivos (archivos, path):
    doc_types = ('.doc', '.docx', '.txt', '.pdf', '.xls', '.ppt', '.xlsx', '.pptx')
    img_types = ('.jpg', '.jpeg', '.png', '.svg', '.gif')
    sw_types = ('.exe', '.py','.ipynb')

    for elem in archivos:
        ruta = os.path.join(path, elem)
    # Saltar carpetas (IMPORTANTE)
        if os.path.isdir(ruta):
            continue
        if elem.endswith(doc_types):
            shutil.move(ruta, os.path.join(path, "Documentos", elem))
        elif elem.endswith(img_types):
            shutil.move(ruta, os.path.join(path, "Imagenes", elem))
        elif elem.endswith(sw_types):
            shutil.move(ruta, os.path.join(path, "SW", elem))
        else:
            shutil.move(ruta, os.path.join(path, "Otros", elem))