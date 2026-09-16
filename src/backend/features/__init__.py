"""One module per operation on the page. Each has the same layout:

    hand-written algorithm   the part the assignment is about
    library baseline         a library doing the same job, for comparison
    view()                   the Flask route: read the upload, run both, render

app.py maps a URL to each module's view().
"""
