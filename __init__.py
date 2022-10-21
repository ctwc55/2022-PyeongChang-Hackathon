from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def mainPage():
    return '<h1>Food detect!</h1>'


if __name__ == '__main__':
    app.run(debug=True, port='8000')