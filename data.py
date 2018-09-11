"""

web script in flask

"""

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from os import listdir
from os.path import isfile, join



def edit_files():
    onlyfiles = [f for f in listdir("no") if isfile(join("no", f))]
    print(onlyfiles)


app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/hello', methods=['GET', 'POST'])


def upload_file_1():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)
    # edit_files()