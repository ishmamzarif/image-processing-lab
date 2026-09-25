/* Live preview for the per-pixel operations: brightness, channels, filters and resize.
 *
 * Each output pixel comes from its own input pixel (or, for pixelate, its own
 * tile), so a whole image takes a few milliseconds and can be redone on every
 * movement of a slider. This does that on the file already picked, before
 * anything is submitted. Apply still posts to the server, where the Python
 * loop runs and Pillow's version is set beside it.
 *
 * The arithmetic is copied from backend/features/brightness.py, channels.py,
 * filters.py and resize.py, operation for operation: the same multiplies and adds in
 * the same order, clipped at the same ends, and truncation (| 0, matching
 * astype(np.uint8)) rather than rounding. What is not the same is the
 * downscale: the canvas shrinks a large photo with its own filter, not
 * Pillow's, so the pixels going in can differ a little even though what is
 * done to each one is identical.
 *
 * Pixels here are one flat list, four bytes each (red, green, blue, alpha),
 * row after row: the pixel at (y, x) starts at index (y * width + x) * 4.
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
    var elDims = document.getElementById('live-dims');
    var elMs = document.getElementById('live-ms');
    var result = document.getElementById('result');
    var filterKind = document.getElementById('filter_kind');

    var srcCtx = srcCv.getContext('2d');
    var outCtx = outCv.getContext('2d');

    var src = null;       // the thumbnail's RGBA bytes, never written to
    var srcWidth = 0;     // and its size
    var srcHeight = 0;
    var out = null;       // the ImageData the preview is written into
    var objectUrl = null;
    var queued = false;

    function signed(v) {
        return v > 0 ? '+' + v : v < 0 ? '−' + (-v) : '0';
    }

    // ---------------------------------------------------------------------
    // The filters, copied from backend/features/filters.py
    // ---------------------------------------------------------------------

    // np.clip(value, 0, 255).astype(np.uint8): clip, then drop the fraction.
    // The | 0 matters: the canvas's byte array would round instead.
    function toByte(v) {
        return v < 0 ? 0 : v > 255 ? 255 : v | 0;
    }

    // gray_of(): how bright a colour looks, ITU-R 601-2 luma
    function grayOf(r, g, b) {
        return 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // sepia_of(): the classic sepia tone, each channel capped at 255
    function sepiaOf(r, g, b) {
        return [
            Math.min(255, 0.393 * r + 0.769 * g + 0.189 * b),
            Math.min(255, 0.349 * r + 0.686 * g + 0.168 * b),
            Math.min(255, 0.272 * r + 0.534 * g + 0.131 * b)
        ];
    }

    // One entry per filter, keyed by the Filter select's value: what the page
    // calls it, the id of its slider, how the slider reads in the meta row,
    // and the loop itself. Each run(s, d, width, height, setting) reads the
    // source bytes s and writes the red, green and blue bytes of d.
    var FILTERS = {
        grayscale: {
            name: 'Grayscale',
            slider: 'gray_amount',
            label: function (v) { return 'amount ' + Math.round(v * 100) + '%'; },
            run: function (s, d, width, height, amount) {
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var i = (y * width + x) * 4;
                        var gray = grayOf(s[i], s[i + 1], s[i + 2]);
                        for (var c = 0; c < 3; c++) {
                            d[i + c] = toByte(s[i + c] + amount * (gray - s[i + c]));
                        }
                    }
                }
            }
        },

        invert: {
            name: 'Invert',
            slider: 'invert_amount',
            label: function (v) { return 'amount ' + Math.round(v * 100) + '%'; },
            run: function (s, d, width, height, amount) {
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var i = (y * width + x) * 4;
                        for (var c = 0; c < 3; c++) {
                            var opposite = 255 - s[i + c];
                            d[i + c] = toByte(s[i + c] + amount * (opposite - s[i + c]));
                        }
                    }
                }
            }
        },

        black_and_white: {
            name: 'Black & white',
            slider: 'threshold',
            label: function (v) { return 'threshold ' + v; },
            run: function (s, d, width, height, threshold) {
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var i = (y * width + x) * 4;
                        var value = grayOf(s[i], s[i + 1], s[i + 2]) >= threshold ? 255 : 0;
                        d[i] = value;
                        d[i + 1] = value;
                        d[i + 2] = value;
                    }
                }
            }
        },

        pixelate: {
            name: 'Pixelate',
            slider: 'block',
            label: function (v) { return v + ' px blocks'; },
            run: function (s, d, width, height, block) {
                // the top-left corner of each tile
                for (var top = 0; top < height; top += block) {
                    for (var left = 0; left < width; left += block) {
                        var bottom = Math.min(top + block, height);
                        var right = Math.min(left + block, width);

                        // add up every pixel in the tile, one channel at a time
                        var total = [0, 0, 0];
                        var y, x, c, i;
                        for (y = top; y < bottom; y++) {
                            for (x = left; x < right; x++) {
                                i = (y * width + x) * 4;
                                for (c = 0; c < 3; c++) total[c] += s[i + c];
                            }
                        }
                        var count = (bottom - top) * (right - left);

                        // then paint the whole tile with the average
                        for (y = top; y < bottom; y++) {
                            for (x = left; x < right; x++) {
                                i = (y * width + x) * 4;
                                for (c = 0; c < 3; c++) d[i + c] = toByte(total[c] / count);
                            }
                        }
                    }
                }
            }
        },

        vintage: {
            name: 'Vintage',
            slider: 'vintage_amount',
            label: function (v) { return 'amount ' + Math.round(v * 100) + '%'; },
            run: function (s, d, width, height, amount) {
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var i = (y * width + x) * 4;
                        var sepia = sepiaOf(s[i], s[i + 1], s[i + 2]);
                        for (var c = 0; c < 3; c++) {
                            d[i + c] = toByte(s[i + c] + amount * (sepia[c] - s[i + c]));
                        }
                    }
                }
            }
        }
    };

    // ---------------------------------------------------------------------
    // Resize, copied from backend/features/resize.py
    // ---------------------------------------------------------------------

    // new_size(): the size after scaling, rounded to whole pixels, at least 1
    function newSize(oldSize, scale) {
        return Math.max(1, Math.floor(oldSize * scale + 0.5));
    }

    // old_positions(): for each new pixel along one side, the old pixel under
    // its middle -- start half a step in, then add one step per pixel
    function oldPositions(oldSize, size) {
        var step = oldSize / size;
        var position = step * 0.5;
        var positions = [];
        for (var i = 0; i < size; i++) {
            positions.push(Math.floor(position));
            position += step;
        }
        return positions;
    }

    // resize_nearest(): every new pixel copies the old pixel under its middle
    function resizeNearest(s, d, oldWidth, oldHeight, scale) {
        var width = newSize(oldWidth, scale);
        var height = newSize(oldHeight, scale);
        var rows = oldPositions(oldHeight, height);
        var columns = oldPositions(oldWidth, width);

        for (var y = 0; y < height; y++) {
            for (var x = 0; x < width; x++) {
                var from = (rows[y] * oldWidth + columns[x]) * 4;
                var to = (y * width + x) * 4;
                d[to] = s[from];
                d[to + 1] = s[from + 1];
                d[to + 2] = s[from + 2];
                d[to + 3] = 255;
            }
        }
    }

    // the value of the chosen filter's slider
    function filterSetting(filter) {
        return parseFloat(document.getElementById(filter.slider).value);
    }

    // ---------------------------------------------------------------------
    // The operations this preview serves
    // ---------------------------------------------------------------------

    // One entry per operation: the radio that selects it, its sliders, what
    // the meta row says, and apply(s, d, p, width, height), which fills d from
    // s (width and height are the source's). p holds the sliders' values in the
    // same order as `sliders`. An operation that changes the picture's size
    // also says what size its output is (outputSize); the rest keep the size.
    var OPS = [
        {
            radio: 'op-brightness',
            title: 'Brightness',
            sliders: ['gain'],
            label: function (p) { return 'gain ' + p[0].toFixed(2) + '×'; },
            // brighten(): the | 0 is doing real work, as in toByte
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
            sliders: ['red', 'green', 'blue'],
            label: function (p) {
                return 'R ' + signed(p[0]) + ' · G ' + signed(p[1]) + ' · B ' + signed(p[2]);
            },
            // shift_channel(): whole numbers throughout, so there is nothing to
            // truncate, only the two ends to clip
            apply: function (s, d, p) {
                for (var i = 0; i < s.length; i += 4) {
                    for (var c = 0; c < 3; c++) {
                        var v = s[i + c] + p[c];
                        d[i + c] = v < 0 ? 0 : v > 255 ? 255 : v;
                    }
                    d[i + 3] = 255;
                }
            }
        },
        {
            radio: 'op-filters',
            title: 'Filters',
            // every filter's slider, so moving any of them redraws; only the
            // chosen filter's is visible, and only its value is used
            sliders: ['gray_amount', 'invert_amount', 'threshold', 'block', 'vintage_amount'],
            label: function () {
                var filter = FILTERS[filterKind.value];
                return filter.name + ' · ' + filter.label(filterSetting(filter));
            },
            apply: function (s, d, p, width, height) {
                var filter = FILTERS[filterKind.value];
                filter.run(s, d, width, height, filterSetting(filter));
                for (var i = 3; i < d.length; i += 4) d[i] = 255;   // RGB only
            }
        },
        {
            radio: 'op-resize',
            title: 'Resize',
            sliders: ['scale'],
            outputSize: function (p, width, height) {
                return [newSize(width, p[0] / 100), newSize(height, p[0] / 100)];
            },
            label: function (p) {
                var size = this.outputSize(p, srcWidth, srcHeight);
                return p[0] + '% · to ' + size[0] + ' × ' + size[1];
            },
            apply: function (s, d, p, width, height) {
                resizeNearest(s, d, width, height, p[0] / 100);
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

    // ---------------------------------------------------------------------
    // Drawing
    // ---------------------------------------------------------------------

    function render() {
        queued = false;
        var op = active();
        if (!src || !op) return;
        var p = op.inputs.map(function (el) { return parseFloat(el.value); });

        // make the preview canvas the size this operation's output will be
        var size = op.outputSize ? op.outputSize(p, srcWidth, srcHeight) : [srcWidth, srcHeight];
        if (out.width !== size[0] || out.height !== size[1]) {
            outCv.width = size[0];
            outCv.height = size[1];
            out = outCtx.createImageData(size[0], size[1]);
        }

        var t0 = performance.now();
        op.apply(src, out.data, p, srcWidth, srcHeight);
        outCtx.putImageData(out, 0, 0);
        elMs.textContent = (performance.now() - t0).toFixed(1) + ' ms';
        elTitle.textContent = op.title;
        elParams.textContent = op.label(p);

        // the colour cube (rgb-cube.js) redraws from the same two images.
        // It pairs pixels by position, which only works while the size is kept.
        section.dispatchEvent(new CustomEvent('live-render', {
            detail: {
                before: src,
                after: out.data,
                pixels: srcWidth * srcHeight,
                sameSize: !op.outputSize
            }
        }));
    }

    // input events arrive faster than the screen redraws, so the work is
    // batched to one pass per frame
    function schedule() {
        if (queued) return;
        queued = true;
        requestAnimationFrame(render);
    }

    // bring the preview up (replacing any result on screen) and redraw it
    function showPreview() {
        if (!src) return;
        section.hidden = false;
        schedule();
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
            srcWidth = w;
            srcHeight = h;
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

    // moving any slider shows the preview straight away
    OPS.forEach(function (op) {
        op.inputs.forEach(function (el) {
            el.addEventListener('input', showPreview);
        });
        // the section is shared, so switching between operations redraws it
        op.radioEl.addEventListener('change', schedule);
    });

    // and so does picking a filter: it is applied as soon as it is chosen
    filterKind.addEventListener('change', showPreview);
})();
