from flask import Flask, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# A simple route that returns a JSON response
@app.route('/api/data')
def get_data():
    return jsonify({"message": "Hello from Flask!"})

# Run the application
if __name__ == '__main__':
    app.run(debug=True)