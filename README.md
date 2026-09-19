**to recreate ->** (Python 3.14) </br>
<code>python -m venv .venv</code></br>
<code>source .venv/bin/activate</code> &nbsp; (mac / linux)</br>
<code>source .venv/Scripts/activate</code> &nbsp; (windows, git bash)</br>
<code>python -m pip install -r requirements.txt</code></br>

**to run ->** </br>
<code>python src/app.py</code></br>
<code>open http://127.0.0.1:5000</code></br>

**select python interpreter ->** </br>
<code>in vscode, Ctrl+Shift+P</code> &nbsp; (Cmd+Shift+P on mac)</br>
<code>Python: Select Interpreter</code>  </br>
<code>choose .venv/bin/python</code> &nbsp; (mac / linux)</br>
<code>choose .venv/Scripts/python.exe</code> &nbsp; (windows)</br>

**where things are ->** </br>
<code>src/app.py</code> &nbsp; entry point, maps each URL to a feature</br>
<code>src/backend/features/</code> &nbsp; one file per operation: blur, sharpen, edges, denoise, compress, encrypt, decrypt, brightness, channels, filters, resize</br>
<code>src/backend/common/</code> &nbsp; helpers shared by the features</br>
<code>src/backend/transforms.py</code> &nbsp; the hand-written FFT / DFT / NTT</br>
<code>src/frontend/templates/</code> &nbsp; the page, split into partials</br>
<code>src/frontend/static/</code> &nbsp; css and js</br>
