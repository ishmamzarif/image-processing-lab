"""One module per operation on the page. Each has the same layout:

    hand-written algorithm   the part the assignment is about
    library baseline         a library doing the same job, for comparison
    view()                   the Flask route: read the upload, run both, render

app.py maps a URL to each module's view().

Encrypt and decrypt are a pair, and each goes one way only: encrypt.py turns a
picture into a ciphertext, decrypt.py turns a ciphertext back into a picture,
and neither imports the other. Everything the two must agree on lives in common/
instead -- phase_keys.py for the masks a passphrase draws, cipher_file.py for
what a ciphertext looks like as a PNG. Encrypt writes the cipher, the stretch
and the crop into the file it hands you, so a saved ciphertext can be opened on
the Decrypt page later with nothing but the passphrase.
"""
