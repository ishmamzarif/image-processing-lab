/* Live preview for the point operations, brightness and channels.
 *
 * In a point operation each output pixel comes from its own input pixel
 * alone, so a whole image takes a millisecond or two and can be redone on
 * every movement of a slider. This does that on the file already picked,
 * before anything is submitted. Apply still posts to the server, where the
 * Python loop runs and Pillow's version is set beside it.
 *
 * The arithmetic is brighten() and shift_channel() from app2.py, operation
 * for operation: the same double multiply or whole-number add, clipped at the
 * same ends, and truncation (| 0, matching astype(np.uint8)) rather than
 * rounding. What is not the same is the downscale: the canvas shrinks a large
 * photo with its own filter, not Pillow's, so the pixels going in can differ
 * a little even though what is done to each one is identical.
 */
(function () {
    'use strict';

    var MAX_DIM = window.MAX_DIM || 500;

    var input = document.getElementById('image');
    var section = document.getElementById('live');
    if (!input || !section) return;

    var srcCv = document.getElementById('live-src');
    var outCv = document.getElementById('live-out');
    var elTitle = document.getElementById('live-title');
    var elParams = document.getElementById('live-params');
    var elFn = document.getElementById('live-fn');
    var elDims = document.getElementById('live-dims');
    var elMs = document.getElementById('live-ms');
    var result = document.getElementById('result');

    var srcCtx = srcCv.getContext('2d');
    var outCtx = outCv.getContext('2d');

    var src = null;       // the thumbnail's RGBA bytes, never written to
    var out = null;       // the ImageData the preview is written into
    var objectUrl = null;
    var queued = false;

    function signed(v) {
        return v > 0 ? '+' + v : v < 0 ? '−' + (-v) : '0';
    }

    // One entry per operation this preview serves: the radio that selects it,
    // its sliders in order, what the meta row says, and the per-value
    // arithmetic from app2.py. p holds the sliders' values in the same order.
    var OPS = [
        {
            radio: 'op-brightness',
            title: 'Brightness',
            fn: 'brighten',
            sliders: ['gain'],
            label: function (p) { return 'gain ' + p[0].toFixed(2) + '×'; },
            // The | 0 is doing real work: a Uint8ClampedArray rounds whatever
            // it is given, and the server truncates.
            apply: function (s, d, p) {
                var g = p[0];
                for (var i = 0; i < s.length; i += 4) {
                    for (var c = 0; c < 3; c++) {
                        var v = s[i + c] * g;
                        d[i + c] = v > 255 ? 255 : v | 0;
                    }
                    d[i + 3] = 255;   // RGB only, as on the server
                }
            }
        },
        {
            radio: 'op-channels',
            title: 'Channels',
            fn: 'shift_channel',
            sliders: ['red', 'green', 'blue'],
            label: function (p) {
                return 'R ' + signed(p[0]) + ' · G ' + signed(p[1]) + ' · B ' + signed(p[2]);
            },
            // whole numbers throughout, so there is nothing to truncate, only
            // the two ends to clip
            apply: function (s, d, p) {
                for (var i = 0; i < s.length; i += 4) {
                    for (var c = 0; c < 3; c++) {
                        var v = s[i + c] + p[c];
                        d[i + c] = v < 0 ? 0 : v > 255 ? 255 : v;
                    }
                    d[i + 3] = 255;
                }
            }
        }
    ];

    OPS.forEach(function (op) {
        op.radioEl = document.getElementById(op.radio);
        op.inputs = op.sliders.map(function (id) { return document.getElementById(id); });
    });

    function active() {
        for (var i = 0; i < OPS.length; i++) {
            if (OPS[i].radioEl.checked) return OPS[i];
        }
        return null;
    }

    function render() {
        queued = false;
        var op = active();
        if (!src || !op) return;
        var p = op.inputs.map(function (el) { return parseFloat(el.value); });
        var t0 = performance.now();
        op.apply(src, out.data, p);
        outCtx.putImageData(out, 0, 0);
        elMs.textContent = (performance.now() - t0).toFixed(1) + ' ms';
        elTitle.textContent = op.title;
        elParams.textContent = op.label(p);
        elFn.textContent = op.fn;
    }

    // input events arrive faster than the screen redraws, so the work is
    // batched to one pass per frame
    function schedule() {
        if (queued) return;
        queued = true;
        requestAnimationFrame(render);
    }

    function load(file) {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(file);

        var im = new Image();
        im.onload = function () {
            // the same cap as img.thumbnail((MAX_DIM, MAX_DIM)), worked out as
            // conv-viz.js does it
            var s = Math.min(1, MAX_DIM / Math.max(im.naturalWidth, im.naturalHeight));
            var w = Math.max(1, Math.round(im.naturalWidth * s));
            var h = Math.max(1, Math.round(im.naturalHeight * s));

            srcCv.width = outCv.width = w;
            srcCv.height = outCv.height = h;
            srcCtx.drawImage(im, 0, 0, w, h);
            src = srcCtx.getImageData(0, 0, w, h).data;
            out = outCtx.createImageData(w, h);

            elDims.textContent = w + ' × ' + h;
            render();
        };
        im.src = objectUrl;
    }

    input.addEventListener('change', function () {
        var f = input.files[0];
        src = null;
        if (!f) {
            section.hidden = true;
            return;
        }
        // A result already on screen stays until a slider is touched: that is
        // the kept file being put back after a run, and the result is the one
        // it just produced. The page's own change handler runs first and has
        // already hidden any result a fresh pick made stale.
        section.hidden = !!(result && !result.hidden);
        load(f);
    });

    OPS.forEach(function (op) {
        op.inputs.forEach(function (el) {
            el.addEventListener('input', function () {
                if (!src) return;
                section.hidden = false;
                schedule();
            });
        });
        // the section is shared, so switching between the two redraws it
        op.radioEl.addEventListener('change', schedule);
    });
})();
