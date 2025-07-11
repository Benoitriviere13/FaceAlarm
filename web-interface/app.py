from flask import Flask, render_template
import os

app = Flask(__name__)

CAM_COUNT = 5

@app.route("/")
def index():
    return render_template("index.html", cam_count=CAM_COUNT)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
