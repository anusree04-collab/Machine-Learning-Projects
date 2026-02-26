from flask import Flask, request, jsonify,render_template
from clustering import run_clustering

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/cluster", methods=["POST"])
def cluster():
    algo = request.json["algorithm"]
    x_pca, labels = run_clustering(algo)

    return jsonify({
        "x": x_pca[:,0].tolist(),
        "y": x_pca[:,1].tolist(),
        "labels": labels.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)

