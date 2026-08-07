from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    name = request.args.get("name", "world")
    s = str(request.args)
    # always return a string or a html template
    return render_template("index.html", name = name, str = s)

if __name__ == "__main__":
    app.run(debug = True)
