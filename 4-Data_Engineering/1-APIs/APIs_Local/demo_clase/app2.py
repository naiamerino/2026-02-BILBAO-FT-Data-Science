
from flask import Flask, request
# ahora vamos a leer y escribir libros de una bbdd
import sqlite3

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "`<h1>`No me gusta que me salga URL not found `</h1><p>`This site is a prototype API for distant reading of science fiction novels.`</p>`"

# El endpoint es igual que antes. Lo que es diferente es lo que hace por debajo
# Ahora se conecta a la base de datos
@app.route('/api/v1/resources/books/all', methods=['GET'])
def get_all():
    # Hacemos la conexión y le hacemos un select all
    connection = sqlite3.connect('books.db')
    cursor = connection.cursor()
    select_books = "SELECT * FROM books"
    result = cursor.execute(select_books).fetchall()
    connection.close()
    return {'books': result}

# Aquí pasamos un string y nos devuelve todos los libros de ese autor
@app.route('/api/v1/resources/book/<string:author>', methods=['GET'])
def get_by_author(author):
    connection = sqlite3.connect('books.db')
    cursor = connection.cursor()
    select_books = "SELECT * FROM books WHERE author=?"
    result = cursor.execute(select_books, (author,)).fetchall()
    connection.close()
    return {'books': result}

# Puedo pasarle id, fecha o autor
@app.route('/api/v1/resources/book/filter', methods=['GET'])
def filter_table():
    query_parameters = request.get_json()
    id = query_parameters.get('id')
    published = query_parameters.get('published')
    author = query_parameters.get('author')
    connection = sqlite3.connect('books.db')
    cursor = connection.cursor()
    query = "SELECT * FROM books WHERE"
    to_filter = [] #voy construyuendo la petición
    if id: # si le he pasado ID
        query += ' id=? AND'
        to_filter.append(id)
    if published: # si le he pasado fecha
        query += ' published=? AND'
        to_filter.append(published)
    if author:
        query += ' author=? AND'
        to_filter.append(author)
        if not (id or published or author):
            return "page not found 404"
    query = query[:-4] + ';'# quito el espacio más AND anterior
    result = cursor.execute(query, to_filter).fetchall()
    connection.close()
    return {'books': result}

app.run()