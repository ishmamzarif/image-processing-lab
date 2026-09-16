"""Image Processing Lab: the web app's entry point.

Run from the repository root:

    python src/app.py            then open http://127.0.0.1:5000

This file only wires things together. The image processing lives in
backend/features/ (one file per operation), the helpers they share in
backend/common/, and the page itself in frontend/.

Every operation works the same way. The page is one form; each "Apply" button
posts it to its own URL. The matching view() reads the uploaded image, runs the
hand-written algorithm and a library version, times both, and renders
index.html again with the results filled in. Nothing is stored between
requests.
"""

from flask import Flask, render_template

from backend.common.limits import MAX_DIM
from backend.common.uploads import UploadError
from backend.features import blur, brightness, channels, compress, denoise, edges, sharpen

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
)

# The template sizes the pre-run preview to match what the processed image will
# be, so it needs the same number. Exposing it here keeps the two in step.
app.jinja_env.globals["MAX_DIM"] = MAX_DIM


@app.route("/")
def index():
    """The empty page, before anything has been run."""
    return render_template("index.html")


# URL -> the feature module whose view() handles it
FEATURES = {
    "/blur": blur,
    "/sharpen": sharpen,
    "/edges": edges,
    "/denoise": denoise,
    "/compress": compress,
    "/brightness": brightness,
    "/channels": channels,
}

for url, feature in FEATURES.items():
    app.add_url_rule(url, endpoint=url.strip("/"), view_func=feature.view, methods=["POST"])


@app.errorhandler(UploadError)
def upload_error(error):
    """A missing or unreadable upload: show the page again with the message."""
    return render_template("index.html", error=str(error))


if __name__ == "__main__":
    app.run(debug=True)
