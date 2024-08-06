html_training = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <title>Image Viewer</title>
    <style>
        body {
            background-color: #ffffff;
            color: #343a40;
            font-family: Arial, sans-serif;
        }
        .container {
            max-width: 1500px;
            margin-top: 50px;
        }
        .header {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-right: 20px;
        }
        #latest-image {
            width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #description {
            margin-top: 20px;
            font-size: 1.2rem;
            text-align: center;
        }
        #timestamp {
            margin-top: 10px;
            font-size: 0.9rem;
            text-align: center;
            color: #6c757d;
        }
        #heartbeat {
            width: 15px;
            height: 15px;
            background-color: red;
            border-radius: 50%;
            position: fixed;
            top: 10px;
            right: 10px;
            opacity: 0;
            transition: opacity 0.2s ease-in-out;
            z-index: 1000;
        }
        #heartbeat.active {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div id="heartbeat"></div>
    <div class="container">
        <div class="header">
            <h1>Image Viewer</h1>
            <a href="/interactive" class="btn btn-primary">Switch to Interactive Viewer</a>
        </div>
        <img id="latest-image" src="" class="img-fluid" alt="Latest Image">
        <p id="description"></p>
        <p id="timestamp"></p>
    </div>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script type="text/javascript">
        var socket = io();
        var heartbeat = document.getElementById('heartbeat');

        function triggerHeartbeat() {
            heartbeat.classList.add('active');
            setTimeout(function() {
                heartbeat.classList.remove('active');
            }, 200);
        }

        socket.on('image_updated', function(data) {
            var arrayBuffer = new Uint8Array(data.image_data.length);
            for (var i = 0; i < data.image_data.length; i++) {
                arrayBuffer[i] = data.image_data.charCodeAt(i);
            }
            var blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
            var url = URL.createObjectURL(blob);
            document.getElementById('latest-image').src = url;
            document.getElementById('description').innerText = data.description;
            document.getElementById('timestamp').innerText = 'Last updated: ' + data.timestamp;
            triggerHeartbeat();
        });
    </script>
</body>
</html>
"""

html_interactive = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <title>Interactive Image Viewer</title>
    <style>
        body {
            background-color: #ffffff;
            color: #343a40;
            font-family: Arial, sans-serif;
        }
        .container {
            max-width: 800px;
            margin-top: 50px;
        }
        .header {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-right: 20px;
        }
        #latest-image {
            width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #description {
            margin-top: 20px;
            font-size: 1.2rem;
            text-align: center;
        }
        #timestamp {
            margin-top: 10px;
            font-size: 0.9rem;
            text-align: center;
            color: #6c757d;
        }
        .form-group {
            margin-top: 20px;
        }
        #heartbeat {
            width: 15px;
            height: 15px;
            background-color: red;
            border-radius: 50%;
            position: fixed;
            top: 10px;
            right: 10px;
            opacity: 0;
            transition: opacity 0.2s ease-in-out;
            z-index: 1000;
        }
        #heartbeat.active {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div id="heartbeat"></div>
    <div class="container">
        <div class="header">
            <h1>Interactive Image Viewer</h1>
            <a href="/" class="btn btn-primary">Switch to Regular Viewer</a>
        </div>
        <img id="latest-image" src="" class="img-fluid" alt="Latest Image">
        <p id="description"></p>
        <p id="timestamp"></p>
        
        <div class="form-group">
            <label for="joint-angles">Joint Angles</label>
            <div id="joint-angles">
                <input type="range" class="form-control-range slider" id="slider01" min="0" max="100" value="50" oninput="sendSliderValues()">
                <input type="range" class="form-control-range slider" id="slider02" min="0" max="100" value="50" oninput="sendSliderValues()">
                <input type="range" class="form-control-range slider" id="slider03" min="0" max="100" value="50" oninput="sendSliderValues()">
                <input type="range" class="form-control-range slider" id="slider04" min="0" max="100" value="50" oninput="sendSliderValues()">
                <input type="range" class="form-control-range slider" id="slider05" min="0" max="100" value="50" oninput="sendSliderValues()">
                <input type="range" class="form-control-range slider" id="slider06" min="0" max="100" value="50" oninput="sendSliderValues()">
                <input type="range" class="form-control-range slider" id="slider07" min="0" max="100" value="50" oninput="sendSliderValues()">
            </div>
        </div>
        <div class="form-group">
            <label for="scale-modifier">Scale Modifier</label>
            <div id="scale-modifier">
                <input type="range" class="form-control-range slider" id="slider08" min="0" max="100" value="100" oninput="sendSliderValues()">
            </div>
        </div>
    </div>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script type="text/javascript">
        var socket = io();
        var heartbeat = document.getElementById('heartbeat');

        function sendSliderValues() {
            var sliders = document.getElementsByClassName('slider');
            var values = {};
            for (var i = 0; i < sliders.length; i++) {
                values['slider' + i] = sliders[i].value;
            }
            socket.emit('slider_update', values);
        }

        function triggerHeartbeat() {
            heartbeat.classList.add('active');
            setTimeout(function() {
                heartbeat.classList.remove('active');
            }, 200);
        }

        socket.on('image_updated', function(data) {
            var arrayBuffer = new Uint8Array(data.image_data.length);
            for (var i = 0; i < data.image_data.length; i++) {
                arrayBuffer[i] = data.image_data.charCodeAt(i);
            }
            var blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
            var url = URL.createObjectURL(blob);
            document.getElementById('latest-image').src = url;
            document.getElementById('description').innerText = data.description;
            document.getElementById('timestamp').innerText = 'Last updated: ' + data.timestamp;
            triggerHeartbeat();
        });
    </script>
</body>
</html>
"""