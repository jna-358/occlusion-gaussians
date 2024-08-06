from flask import Flask, request, render_template_string, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime
from html import html_training, html_interactive

app = Flask(__name__)
socketio = SocketIO(app)
latest_image = None
latest_description = None
latest_timestamp = None

# Keep the latest slider values in memory
slider_values = {
  "slider01": "50",
  "slider02": "50",
  "slider03": "50",
  "slider04": "50",
  "slider05": "50",
  "slider06": "50",
  "slider07": "50",
  "slider08": "100"
}

def log(msg):
    with open('/content/webserver/log.txt', 'a') as f:
        f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')

@app.route('/upload', methods=['POST'])
def upload_file():
    global latest_image, latest_description, latest_timestamp
    if 'file' not in request.files or 'description' not in request.form:
        return "No file or description part", 400
    file = request.files['file']
    description = request.form['description']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        latest_image = file.read()
        latest_description = description
        latest_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        socketio.emit('image_updated', {
            'image_data': latest_image.decode('latin1'),
            'description': latest_description,
            'timestamp': latest_timestamp
        })
        return "File uploaded successfully", 200
    return "Invalid file type", 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template_string(html_training)

@app.route('/interactive')
def interactive():
    return render_template_string(html_interactive)


@socketio.on('slider_update')
def handle_slider_update(data):
    log(f"Received slider update: {data}")
    global slider_values
    slider_values = data

@app.route('/get_params')
def get_params():
    return jsonify(slider_values)

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
