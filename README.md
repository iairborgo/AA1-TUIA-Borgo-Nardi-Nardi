# AA1-TUIA-Borgo-Nardi-Nardi


### Para su uso:
Descarga de Imagen: 
        docker pull iairborgo/aa_borgonardix2:latest

Correr imagen:
        docker run -v "path/a/archivo:/app/data"  iairborgo/aa-borgonardix2 --f < app/data/nombre_archivo.csv >

Uso del script python:
    python inferencia.py --f < nombre_archivo >

path/a/archivo es una ruta a la carpeta donde tendremos .csv en el que se deberia tener los datos para los dias que se quieren predecir,
de la misma forma en la que los datos originales fueron presentados.

Luego inferencia.py se encarga de realizar las transformaciones necesarias para su posterior inferencia, y guarda los resultados
en la misma carpeta que path/a/archivo