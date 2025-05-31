# backend.py
from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/data')
def get_data():
    df = pd.read_csv('your_file.csv')  # Replace with actual path
    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
