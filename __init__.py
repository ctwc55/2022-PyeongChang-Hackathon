from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def mainPage():
    return render_template('index.html')

@app.route('/<name>')
def otherPages(name):
    return render_template(name)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')