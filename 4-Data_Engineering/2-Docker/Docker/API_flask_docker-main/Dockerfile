# DOCKERFILE PARA DOCKERIZAR MI APLICACIÓN (APP.PY)

# Le estoy indicando que vaya a dockerfile y busque esta versión de python
# Pongo que tenga mi versión. Si funciona en mi ordenador
FROM python:3.11.15-alpine3.23 
# Crea la carpeta src
RUN mkdir /src
WORKDIR /src
# copia todo lo que haya en mi directorio a esa carpeta src
ADD . /src
RUN mkdir /src/tmp
RUN touch /src/tmp/data.txt
RUN echo "[{\"name\":\"alejandru\",\"email\":\"alejandru@gmail.com\"},{\"name\":\"birja\",\"email\":\"birja@gmail.com\"},{\"name\":\"muchelle\",\"email\":\"muchelle@gmail.com\"},{\"name\":\"christianu\",\"email\":\"christianu@gmail.com\"},{\"name\":\"alvaru\",\"email\":\"alvaru@gmail.com\"}]" >> /src/tmp/data.txt
# que instale todo lo que necesite. El requirements contendrá todo
# lo que sea necesario
RUN pip install -r requirements.txt
# Comando para ejecutar desde terminal: ejecutar python app.py
CMD ["python", "app.py"]
# que deje abierto el puerto 5000 porque es donde se ejecuta Flask
EXPOSE 5000