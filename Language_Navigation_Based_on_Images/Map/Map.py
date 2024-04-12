import atexit
import shutil
import os
import time

import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_session import Session
from GPT import DialogSystem
from pathGeneration import DirectedGraph
from RelativePosition import RelativePosition
from Detector import Detector
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Define the directory for log files
log_directory = 'logfiles'
os.makedirs(log_directory, exist_ok=True)  # Make sure the catalogue exists

# Create a unique log file name using the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_directory, f'log_{current_time}.log')

# Create a logger and set the log level
logging.basicConfig(level=logging.INFO)

# Create a rotating file handler to limit the size of the log file and set the number of backups
file_handler = RotatingFileHandler(log_file_path, maxBytes=10240, backupCount=10)

# Creating a Log Format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)


# Flask application setup with session management configuration.
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session'
# Add the above log handler to your Flask application
app.logger.addHandler(file_handler)

# Make sure the session file directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
app.config["SESSION_PERMANENT"] = False
Session(app)
os.environ["OMP_NUM_THREADS"] = "6"


graph = None
graph_dict = None
end_node_id = None
# Define vertex data: vertex ID, image path, whether it is an intersection or not

vertices_info = [
    ('Home', 'home_junction.png', True,
     {'junction_coord': (1004, 971), 'forward_exits_coords': (1004, 745), 'left_exit_coords': (353, 971)}),
    ('Moon Street', 'home_straight.png', False),
    ('Heriot-Watt University', 'school.png', False),
    ('Princesses Street', 'home_left.png', False),
    ('Cross Junction', 'junction.png', True,
     {'junction_coord': (1135, 921), 'forward_exits_coords': (1135, 700), 'left_exit_coords': (40, 921), 'right_exit_coords': (2000, 921)}),
    ('Queen Street', 'junction_straight.png', False),
    ('McDonald\'s', 'McDonald\'s.png', False),
    ('Princes Street', 'junction_left.png', False),
    ('St.James Quarter','St.James_Quarter.png', False),
    ('King Street', 'junction_right.png', False),
    ('Sainsbury','Sainsbury.png', False),
]

edges_info = [
    ('Home', 'Moon Street', 'forward', 1),
    ('Moon Street', 'Home', 'back', 1),
    ('Moon Street', 'Heriot-Watt University', 'forward', 1),
    ('Heriot-Watt University', 'Moon Street',  'back', 1),
    ('Home', 'Princesses Street', 'left', 1),
    ('Princesses Street', 'Home', 'back', 1),
    ('Princesses Street', 'Cross Junction', 'forward', 1),
    ('Cross Junction', 'Princesses Street', 'back', 1),
    ('Cross Junction', 'Queen Street', 'forward', 1),
    ('Queen Street', 'Cross Junction', 'back', 1),
    ('Cross Junction', 'Princes Street', 'left', 1),
    ('Princes Street', 'Cross Junction', 'back', 1),
    ('Cross Junction', 'King Street', 'right', 1),
    ('King Street', 'Cross Junction', 'back', 1),
    ('Queen Street', 'McDonald\'s', 'forward', 1),
    ('McDonald\'s', 'Queen Street', 'back', 1),
    ('Princes Street', 'St.James Quarter', 'forward', 1),
    ('St.James Quarter', 'Princes Street', 'back', 1),
    ('King Street', 'Sainsbury', 'forward', 1),
    ('Sainsbury', 'King Street', 'back', 1),
]

items_list_cache = {}

def initialize_graph():
    """
    Initializes the directed graph for navigation, constructs the graph based on predefined vertices and edges, and
    sets up the Detector and RelativePosition instances for object detection and relative positioning.
    """
    global graph
    global graph_dict
    global relativePosition
    global detector
    detector = Detector()
    graph = DirectedGraph()
    graph.construct_graph(vertices_info, edges_info)
    graph_dict = graph.convert_directed_graph_to_dict(graph)
    relativePosition = RelativePosition()


@app.route('/', methods=['GET'])
def start():
    """
    Route to render the starting page of the web application.
    """
    return render_template('start.html')


@app.route('/set_start_end', methods=['POST'])
def set_start_node():
    """
    Route to set the starting and ending points for navigation based on user input.
    """
    global end_node_id
    start_node_id = request.form['start_node_id']
    end_node_id = request.form['end_node_id']
    app.logger.info(f'User starts from {start_node_id} to destination {end_node_id}')
    return redirect(url_for('index', node_id=start_node_id))


@app.route('/index', methods=['GET', 'POST'])
def index():
    """
    Main route to display the current node and available navigation options.
    """
    node_id = request.args.get('node_id', 'A')
    node = graph_dict.get(node_id, 'A')
    return render_template('index.html', node=node, node_id=node_id)


@app.route('/navigate', methods=['POST'])
def navigate():
    """
    API route to process navigation requests and respond with information about the next node.
    """
    direction = request.json['direction']
    current_node_id = request.json['current_node_id']
    current_node = graph_dict.get(current_node_id)

    app.logger.info(f'User choose Direction: {direction}')

    next_node_id = None
    new_image_path = None
    for edge in current_node['edges']:
        if edge['direction'] == direction:
            next_node_id = edge['to']
            break

    if not next_node_id:
        return jsonify({"error": "No next node found."}), 404

    next_node = graph_dict.get(next_node_id)

    # Returns information about the new node and new direction options
    return jsonify({
        "newNodeId": next_node_id,
        "newImagePath": url_for('static', filename=next_node['image_path']),
        "edges": next_node['edges']
    })

@app.route('/send_message', methods=['POST'])
def send_message():
    """
    API route to process messages sent by the user, interact with the DialogSystem, and return responses.
    """
    start_time = time.time()
    global instruction
    data = request.json
    print(data)
    input_message = data['message']
    current_node_id = data['current_node_id']
    current_node = graph_dict[current_node_id]
    image_path = current_node['image_path']
    app.logger.info(f'User arrived at {current_node_id}')

    # Check if the result for the current image path is cached to avoid reprocessing.
    if image_path not in items_list_cache:
        image = cv2.imread('static/' + image_path)
        # Process new image paths and caching the results
        items_list_cache[image_path] = detector.resultToList(image)
    # Get results from the cache
    items_list = items_list_cache[image_path]

    # Determine the next path based on the current and end nodes.
    path = graph.find_shortest_path(graph_dict, current_node_id, end_node_id)
    if current_node_id == end_node_id:
        instruction = 'You have arrived at destination.'
    else:
        if path['direction'] == 'left' or path['direction'] == 'right':
            relativePosition.parameterAssignment(current_node)
            relativePosition.positionClassification(items_list)
            instruction = relativePosition.turn_description(path['direction'])
        elif path['direction'] == 'forward':
            instruction = 'go forward to next stop.'
        elif path['direction'] == 'back':
            instruction = 'go back to last stop.'

    # Session handling for dialog system state and current path.
    # Logic to either update the session with the new path or retrieve the existing dialog system state.
    if 'current_path' not in session or path != session['current_path']:
        dialog_system = DialogSystem()
        session['dialog_system'] = dialog_system
        session['current_path'] = path
    else:
        dialog_system = session['dialog_system']

    session.modified = True

    # Pass in the user input and pass the parameters to the format_prompt function to modify the prompt.
    dialog_system.add_user_message(input_message)
    dialog_system.format_prompt(items_list, path, instruction)

    app.logger.info(f'User inputs: {input_message}')

    response_message = dialog_system.get_response() # Retrieve response from the dialog system
    end_time = time.time()
    print(f"{end_time - start_time} s")
    app.logger.info(f'System responses: {response_message}')

    return jsonify({"message": response_message})

def cleanup_sessions():
    """
    Cleans up session files stored in the file system to maintain application hygiene.
    """
    folder = app.config['SESSION_FILE_DIR']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


@atexit.register
def cleanup():
    """
    Registered function to clean up resources when the application exits.
    """
    cleanup_sessions()
    print("Session files cleaned up.")


if __name__ == '__main__':
    initialize_graph()
    app.run(debug=True)
