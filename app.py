from flask import Flask, request, render_template
from utilities import get_tryon

@app.route('/',methods=["POST","GET"])
def method_name():
    return "Server for Virtual Try On (Amazon HackOn)"

from utilities import get_tryon

@app.route('/generate', methods=["POST","GET"])
def process_input():
    if request.method=='POST':
        person = request.files['person']
        attire = request.files['attire']
        print('Files received, processing output')
        output = get_tryon(person, attire)
        print("Processed try-on")
        return render_template('home.html',output)