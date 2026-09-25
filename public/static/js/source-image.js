/* The picked image: its name, a preview before any run, and keeping it across runs.
 *
 * Every run is a full-page POST, and a file input always comes back empty after
 * a reload. So the picked File is kept in IndexedDB (database "lab", store
 * "source") and put back into the input when the page loads. The server never
 * knows: it simply receives the same upload again next time.
 *
 * It is only put back on the page an Apply button opened. Any other load --
 * refreshing, a new tab, typing the address -- is a fresh start, exactly like
 * entering the site for the first time: no image, and the kept copy is
 * forgotten. Apply leaves a one-time note in sessionStorage to tell the two
 * apart.
 */
(function () {
    var input = document.getElementById('image');
    var name = document.getElementById('filename');
    var preview = document.getElementById('preview');
    var previewImg = document.getElementById('preview-img');
    var previewDims = document.getElementById('preview-dims');
    var result = document.getElementById('result');
    var empty = document.getElementById('empty');
    var clearBtn = document.getElementById('image-reset');
    var objectUrl = null;
    var restoring = false;

    // Name the chosen file, and show it in the canvas straight away. Any result
    // on screen was computed from a previous pick, so it goes -- unless this is
    // the kept file being put back after a run, when the result on screen is
    // the one it just produced.
    input.addEventListener('change', function () {
        var f = input.files[0];
        name.textContent = f ? f.name : 'No file chosen';
        name.classList.toggle('is-set', !!f);
        clearBtn.hidden = !f;
        if (!restoring) keep(f);

        if (objectUrl) URL.revokeObjectURL(objectUrl);

        if (!f) {
            preview.hidden = true;
            if (empty) empty.hidden = false;
            return;
        }

        if (restoring && result) return;

        objectUrl = URL.createObjectURL(f);
        previewImg.src = objectUrl;
        previewDims.textContent = '—';

        previewImg.onload = function () {
            previewDims.textContent =
                previewImg.naturalWidth + ' × ' + previewImg.naturalHeight;
        };

        preview.hidden = false;
        if (result) result.hidden = true;
        if (empty) empty.hidden = true;
    });

    // localStorage will not do for this: it holds strings only, and a photo as
    // a data URI is past its quota. Without IndexedDB, or without DataTransfer
    // to refill the input, it falls back to picking the file again.
    function store(mode, fn) {
        try {
            var open = indexedDB.open('lab', 1);
            open.onupgradeneeded = function () { open.result.createObjectStore('source'); };
            open.onsuccess = function () {
                var tx = open.result.transaction('source', mode);
                tx.oncomplete = tx.onabort = function () { open.result.close(); };
                fn(tx.objectStore('source'));
            };
        } catch (e) { }
    }

    function keep(f) {
        store('readwrite', function (s) { if (f) s.put(f, 'image'); else s.delete('image'); });
    }

    // Apply leaves this note just before the page goes away. The next page reads
    // it once and removes it, so only that page carries the image over.
    var CARRY = 'carry-image';

    document.getElementById('rail').addEventListener('submit', function () {
        try { sessionStorage.setItem(CARRY, 'yes'); } catch (e) { }
    });

    // Was this page opened by Apply? If sessionStorage cannot be used at all
    // (some private windows), carry the image over anyway, as before.
    function openedByApply() {
        try {
            var carry = sessionStorage.getItem(CARRY) === 'yes';
            sessionStorage.removeItem(CARRY);
            return carry;
        } catch (e) {
            return true;
        }
    }

    var carryImage = openedByApply();

    // A result page's address is the Apply URL (/blur, /filters, ...), and
    // refreshing it would send the form again. Pointing the address back at
    // the site itself makes a refresh a fresh visit instead.
    if (location.pathname !== '/') {
        try { history.replaceState(null, '', '/'); } catch (e) { }
    }

    // after DOMContentLoaded, so that conv-viz.js and adjust.js are listening
    // for the change event too
    document.addEventListener('DOMContentLoaded', function () {
        if (!carryImage) {
            keep(null);   // a fresh start: forget the kept copy
            return;
        }
        store('readonly', function (s) {
            var get = s.get('image');
            get.onsuccess = function () {
                // nothing kept, or the user picked a file while this was loading
                if (!get.result || input.files.length) return;
                try {
                    var dt = new DataTransfer();
                    dt.items.add(get.result);
                    input.files = dt.files;
                } catch (e) { return; }
                restoring = true;
                input.dispatchEvent(new Event('change'));
                restoring = false;
            };
        });
    });

    // the sidebar's Reset: empty the input, which also forgets the kept copy
    clearBtn.addEventListener('click', function () {
        input.value = '';
        input.dispatchEvent(new Event('change'));
    });
})();
