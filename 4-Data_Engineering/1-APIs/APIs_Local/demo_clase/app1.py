import flask
from flask import request, jsonify

books = [
    {'id': 0,
    'title': 'A Fire Upon the Deep',
    'author': 'Vernor Vinge',
    'first_sentence': 'The coldsleep itself was dreamless.',
    'year_published': '1992'},
    {'id': 1,'title': 'The Ones Who Walk Away From Omelas',
    'author': 'Ursula K. Le Guin',
    'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
    'published': '1973'},
    {'id': 2,
    'title': 'Dhalgren',
    'author': 'Samuel R. Delany',
    'first_sentence': 'to wound the autumnal city.',
    'published': '1975'},
    {'id': 3,
    'title': 'The Chain',
    'author': 'Jaime G. Páramo',
    'first_sentence': 'There were tears on her eyes and fears trapped her mind but, inside, the courage of those who have nothing to lose and all to win, flown wild and free.',
    'published': '2025'}
    ]

# Es una configuración para que quede escuchando. 
app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Aquí estoy usando un decorador de Flask. app es un objeto de flask
@app.route('/', methods=['GET'])
def home():
    return "`<h1>`Distant Reading Archive `</h1><p>`This site is a prototype API for distant reading of science fiction novels.`</p>`"

# El endpoint es este: (/api/v1/resources/books/all). Que cuando se ejecute con el metodo GET
# haga lo que indica la función (en este caso devolver books)
@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    return jsonify(books)

#Otro endpoint. Se le pasa un id. Comprueba si hay id. Si está en la petición sigue, si no, error
# Luego comprueba si está en nuestro diccionario y si es que si, devuelve ese libro
@app.route('/api/v1/resources/book', methods=['GET'])
def api_id():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    results = []

    for book in books:
        if book['id'] == id:
            results.append(book)

    return jsonify(results)

# un nuevo endpoint que devuelve datos de un libro por título
@app.route('/api/v1/resources/book/<string:title>', methods=['GET'])
def get_by_title(title):
    for book in books:
        if book['title'] == title:
            return jsonify(book)
    return jsonify({'message': "Book not found"})

# Está en v2 porque es algo diferente. Aquí en vez de un id o título recibe un json
# No puedo hacer la peticion por web porque tengo que mandarle el json en el body
@app.route('/api/v2/resources/book', methods=['GET'])
def get_by_id():
    id = int(request.get_json()['id'])
    for book in books:
        if book['id'] == id:
            return jsonify(book)
    return jsonify({'message': "Book not found"})

# Añadir un libro. Se lo paso además en un json. En este caso solo lo estamos guardando en la variable
# No lo estoy guardando en ninguna parte
@app.route('/api/v1/resources/book', methods=['POST'])
def post_book():
    data = request.get_json()
    books.append(data)
    return data
#siempre al final. es lo que mantiene funcionando el script permanentemente
app.run()