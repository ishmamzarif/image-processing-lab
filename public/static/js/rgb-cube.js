/* The colour cube for Channels and Filters.
 *
 * Every colour is a point in a cube whose three axes are red, green and blue,
 * 0 to 255 each: black in one corner, white in the opposite one, and every
 * grey on the diagonal between them. A sample of the image's pixels is drawn
 * there, each in its own colour, before and after the operation, which turns
 * what the operation does into a shape:
 *
 *   channel offsets   the whole cloud slides, and piles up against the walls
 *   grayscale         the cloud collapses onto the grey diagonal
 *   invert            the cloud is mirrored through the centre of the cube
 *   black & white     everything lands on two corners
 *   pixelate          the cloud thins out: many pixels now share one colour
 *   vintage           the cloud is squeezed towards one brown line
 *
 * It appears twice. In the live preview (partials/live_adjust.html) adjust.js
 * reports every redraw with a "live-render" event, so the cube follows the
 * sliders. On a result page (partials/results/comparison.html) it reads the
 * original and the processed image straight off the page.
 */
(function () {
    'use strict';

    if (!window.Plot3D) return;

    var SAMPLES = 1500;

    // One pixel from each of `count` equal stretches of the image, at a
    // position fixed by a seeded generator. Spread evenly, so no region is
    // left out, and the same every time, so the cloud does not flicker while
    // a slider moves.
    function samplePixels(count, total) {
        var n = Math.min(count, total);
        var picks = new Uint32Array(n);
        var seed = 12345;
        for (var i = 0; i < n; i++) {
            seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
            var lo = Math.floor(i * total / n);
            var hi = Math.floor((i + 1) * total / n);
            picks[i] = lo + seed % Math.max(1, hi - lo);
        }
        return picks;
    }

    // a channel value 0..255 to a world coordinate -1..1
    function w(v) { return v / 127.5 - 1; }

    // the cube's 12 edges, as pairs of corners given as 0/1 per channel
    var EDGES = [
        [[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]],
        [[1, 1, 1], [0, 1, 1]], [[1, 1, 1], [1, 0, 1]], [[1, 1, 1], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 0]], [[1, 0, 0], [1, 0, 1]], [[0, 1, 0], [1, 1, 0]],
        [[0, 1, 0], [0, 1, 1]], [[0, 0, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 1]]
    ];

    // each axis: its name, the corner it runs to from black, and its colour
    var AXES = [['R', [1, 0, 0], '#e0301e'], ['G', [0, 1, 0], '#1f9d3a'], ['B', [0, 0, 1], '#1f5fe0']];

    function makeCube(canvas, modeBox, countEl) {
        var before = null;      // sampled colours, three bytes a pixel
        var after = null;
        var picks = null;
        var pickedFrom = 0;     // the pixel count picks was made for
        var mode = 'both';

        var plot = Plot3D(canvas, { size: 440, yaw: -0.85, pitch: 0.42, draw: draw });

        function corner(c, view) {
            return view.project(c[0] * 2 - 1, c[1] * 2 - 1, c[2] * 2 - 1);
        }

        function drawFrame(ctx, view) {
            // the edges, faint: the frame, not the subject
            ctx.strokeStyle = view.ink;
            ctx.globalAlpha = 0.18;
            ctx.lineWidth = 1;
            ctx.beginPath();
            EDGES.forEach(function (e) {
                var a = corner(e[0], view), b = corner(e[1], view);
                ctx.moveTo(a[0], a[1]);
                ctx.lineTo(b[0], b[1]);
            });
            ctx.stroke();

            // the grey diagonal, black to white, which grayscale collapses onto
            var k = corner([0, 0, 0], view), wt = corner([1, 1, 1], view);
            ctx.setLineDash([3, 4]);
            ctx.globalAlpha = 0.35;
            ctx.beginPath();
            ctx.moveTo(k[0], k[1]);
            ctx.lineTo(wt[0], wt[1]);
            ctx.stroke();
            ctx.setLineDash([]);

            // the three axes out of black, each in its own colour
            ctx.globalAlpha = 1;
            ctx.lineWidth = 2;
            AXES.forEach(function (a) {
                var end = corner(a[1], view);
                ctx.strokeStyle = a[2];
                ctx.beginPath();
                ctx.moveTo(k[0], k[1]);
                ctx.lineTo(end[0], end[1]);
                ctx.stroke();
            });
        }

        // the axes' names, drawn last so the cloud cannot cover them
        function drawLabels(ctx, view) {
            ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            AXES.forEach(function (a) {
                var tip = view.project(a[1][0] * 2.24 - 1, a[1][1] * 2.24 - 1, a[1][2] * 2.24 - 1);
                ctx.lineWidth = 3;
                ctx.strokeStyle = view.bg;
                ctx.strokeText(a[0], tip[0], tip[1]);
                ctx.fillStyle = a[2];
                ctx.fillText(a[0], tip[0], tip[1]);
            });
        }

        // project the sampled colours and sort them far to near
        function projected(colours, view) {
            var n = colours.length / 3;
            var pts = new Array(n);
            for (var i = 0; i < n; i++) {
                var r = colours[i * 3], g = colours[i * 3 + 1], b = colours[i * 3 + 2];
                var p = view.project(w(r), w(g), w(b));
                pts[i] = {
                    x: p[0], y: p[1], d: p[2],
                    c: 'rgb(' + r + ',' + g + ',' + b + ')',
                    pale: 0.299 * r + 0.587 * g + 0.114 * b > 190
                };
            }
            pts.sort(function (a, b) { return b.d - a.d; });
            return pts;
        }

        // A point is its own colour. A pale one also gets a faint dark rim, or
        // it would vanish against the white page; the rest go without, since
        // rims on a dense cloud of mid-tones would darken the whole of it.
        function drawPoints(ctx, view, colours, size, alpha, rim) {
            var pts = projected(colours, view);
            var half = size / 2;
            for (var i = 0; i < pts.length; i++) {
                var p = pts[i];
                if (rim && p.pale) {
                    ctx.globalAlpha = 0.28 * alpha;
                    ctx.fillStyle = view.ink;
                    ctx.fillRect(p.x - half - 1, p.y - half - 1, size + 2, size + 2);
                }
                ctx.globalAlpha = alpha;
                ctx.fillStyle = p.c;
                ctx.fillRect(p.x - half, p.y - half, size, size);
            }
            ctx.globalAlpha = 1;
        }

        // one faint line per pixel, from where it was to where it went
        function drawTrails(ctx, view) {
            ctx.strokeStyle = view.ink;
            ctx.globalAlpha = 0.1;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (var i = 0; i < before.length; i += 3) {
                var a = view.project(w(before[i]), w(before[i + 1]), w(before[i + 2]));
                var b = view.project(w(after[i]), w(after[i + 1]), w(after[i + 2]));
                ctx.moveTo(a[0], a[1]);
                ctx.lineTo(b[0], b[1]);
            }
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        function draw(ctx, view) {
            drawFrame(ctx, view);
            if (before) drawCloud(ctx, view);
            drawLabels(ctx, view);
        }

        function drawCloud(ctx, view) {
            if (mode === 'before') {
                drawPoints(ctx, view, before, 3, 1, true);
            } else if (mode === 'after') {
                drawPoints(ctx, view, after, 3, 1, true);
            } else {
                // where each pixel started, dimmed, then where it ended up
                drawTrails(ctx, view);
                drawPoints(ctx, view, before, 2, 0.3, false);
                drawPoints(ctx, view, after, 3, 1, true);
            }
        }

        modeBox.addEventListener('click', function (e) {
            var btn = e.target.closest('button[data-mode]');
            if (!btn) return;
            mode = btn.dataset.mode;
            modeBox.querySelectorAll('button').forEach(function (b) {
                b.classList.toggle('is-on', b === btn);
            });
            plot.redraw();
        });

        // src and out are RGBA byte arrays of the same image size
        function set(src, out, pixels) {
            if (pixels !== pickedFrom) {
                picks = samplePixels(SAMPLES, pixels);
                pickedFrom = pixels;
                before = new Uint8Array(picks.length * 3);
                after = new Uint8Array(picks.length * 3);
            }
            for (var i = 0; i < picks.length; i++) {
                var o = picks[i] * 4;
                for (var c = 0; c < 3; c++) {
                    before[i * 3 + c] = src[o + c];
                    after[i * 3 + c] = out[o + c];
                }
            }
            countEl.textContent = picks.length.toLocaleString() + ' pixels';
            plot.redraw();
        }

        return { set: set };
    }

    // --- the live preview: follows adjust.js ---
    var live = document.getElementById('live');
    var liveCanvas = document.getElementById('live-cube');
    if (live && liveCanvas) {
        var liveCube = makeCube(liveCanvas,
            document.getElementById('live-cube-mode'), document.getElementById('live-cube-n'));
        live.addEventListener('live-render', function (e) {
            var d = e.detail;
            if (d.sameSize) liveCube.set(d.before, d.after, d.pixels);
        });
    }

    // --- a result page: the original and the output, read off the page ---
    var resultCanvas = document.getElementById('result-cube');
    var original = document.getElementById('result-original');
    var output = document.getElementById('result-output');
    if (resultCanvas && original && output) {
        var resultCube = makeCube(resultCanvas,
            document.getElementById('result-cube-mode'), document.getElementById('result-cube-n'));

        var pixelsOf = function (img) {
            var cv = document.createElement('canvas');
            cv.width = img.naturalWidth;
            cv.height = img.naturalHeight;
            var ctx = cv.getContext('2d');
            ctx.drawImage(img, 0, 0);
            return ctx.getImageData(0, 0, cv.width, cv.height).data;
        };

        var ready = function (img) { return img.complete && img.naturalWidth > 0; };

        var fill = function () {
            if (!ready(original) || !ready(output)) return;
            var n = original.naturalWidth * original.naturalHeight;
            if (n !== output.naturalWidth * output.naturalHeight) return;
            resultCube.set(pixelsOf(original), pixelsOf(output), n);
        };

        original.addEventListener('load', fill);
        output.addEventListener('load', fill);
        fill();
    }
})();
