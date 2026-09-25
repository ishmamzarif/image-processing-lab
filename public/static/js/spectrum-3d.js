/* The Denoise spectrum as a surface: log |X| of the luma, centred, as height.
 *
 * The flat plate above it shows the same spectrum as brightness, where a noise
 * peak is one pale dot among many. As height it is a spike standing out of a
 * low field, which is the whole idea of the notch filter made visible. The
 * picture itself is the mountain in the middle; stripes are the isolated
 * spikes around it; grain is the raised floor everywhere.
 *
 *   Noisy      the spectrum the filter was given, with every bin it removes
 *              tinted blue, as on the flat plate
 *   Filtered   the same spectrum after the mask, on the same height scale, so
 *              what was removed has dropped to the floor
 *
 * The heights come from the server (spectrum_surface in
 * backend/common/spectrum.py), already pooled to at most 64 x 64 and scaled to
 * 0..1, as JSON inside partials/results/denoise.html.
 */
(function () {
    'use strict';

    var canvas = document.getElementById('spectrum-3d');
    var source = document.getElementById('spectrum-3d-data');
    var modeBox = document.getElementById('spectrum-3d-mode');
    if (!canvas || !source || !window.Plot3D) return;

    var data = JSON.parse(source.textContent);
    var rows = data.before.length;
    var cols = data.before[0].length;
    var mode = 'before';

    var BASE = -0.5;       // world height of the floor
    var HEIGHT = 1.05;     // world height of the tallest bin

    // the longer side spans -1..1 and the shorter keeps the spectrum's shape
    var spanX = cols >= rows ? 1 : cols / rows;
    var spanY = rows >= cols ? 1 : rows / cols;
    function wx(c) { return (c / Math.max(1, cols - 1) * 2 - 1) * spanX; }
    function wy(r) { return (1 - r / Math.max(1, rows - 1) * 2) * spanY; }   // row 0 at the back

    // "#0a84ff" -> [10, 132, 255]
    function rgbOf(hex, fallback) {
        var m = /^#([0-9a-f]{6})$/i.exec(hex);
        if (!m) return fallback;
        var n = parseInt(m[1], 16);
        return [n >> 16, (n >> 8) & 255, n & 255];
    }

    // Each quad's colour depends on the data and the light, not on the view,
    // so it is worked out once per mode rather than on every frame. Grey
    // darkens with height, and a light from the upper left shades the slopes
    // so the shape reads; a removed bin takes the accent instead.
    var colours = {};

    function shadeAll(ink, accent) {
        var dx = 2 * spanX / Math.max(1, cols - 1);
        var dy = 2 * spanY / Math.max(1, rows - 1);
        var lx = -0.45, ly = 0.5, lz = 0.74;              // towards the light, unit length

        ['before', 'after'].forEach(function (which) {
            var z = data[which];
            var out = [];
            for (var r = 0; r < rows - 1; r++) {
                for (var c = 0; c < cols - 1; c++) {
                    var h = (z[r][c] + z[r][c + 1] + z[r + 1][c] + z[r + 1][c + 1]) / 4;

                    // the slope across the quad, and its normal (-dz/dx, -dz/dy, 1)
                    var sx = ((z[r][c + 1] - z[r][c]) + (z[r + 1][c + 1] - z[r + 1][c])) / 2 * HEIGHT / dx;
                    var sy = -((z[r + 1][c] - z[r][c]) + (z[r + 1][c + 1] - z[r][c + 1])) / 2 * HEIGHT / dy;
                    var len = Math.sqrt(sx * sx + sy * sy + 1);
                    var light = Math.max(0, (-sx * lx - sy * ly + lz) / len);
                    var shade = 0.6 + 0.4 * light;

                    var removed = which === 'before' && Math.min(
                        data.mask[r][c], data.mask[r][c + 1], data.mask[r + 1][c], data.mask[r + 1][c + 1]) < 0.5;

                    var rgb;
                    if (removed) {
                        rgb = accent.map(function (v) { return Math.round(v * shade); });
                    } else {
                        var g = Math.round((228 - 178 * h) * shade);
                        rgb = [g, g, g];
                    }
                    out.push('rgb(' + rgb.join(',') + ')');
                }
            }
            colours[which] = out;
        });
    }

    var screen = new Float64Array(rows * cols * 3);   // projected vertices
    var depth = new Float64Array((rows - 1) * (cols - 1));
    var order = [];
    for (var q = 0; q < (rows - 1) * (cols - 1); q++) order.push(q);

    function draw(ctx, view) {
        if (!colours.before) shadeAll(rgbOf(view.ink, [0, 0, 0]), rgbOf(view.accent, [10, 132, 255]));
        var z = data[mode];

        // every vertex once
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                var p = view.project(wx(c), wy(r), BASE + z[r][c] * HEIGHT);
                var i = (r * cols + c) * 3;
                screen[i] = p[0];
                screen[i + 1] = p[1];
                screen[i + 2] = p[2];
            }
        }

        // then the quads, back to front
        for (var qi = 0; qi < depth.length; qi++) {
            var qr = (qi / (cols - 1)) | 0, qc = qi % (cols - 1);
            var a = (qr * cols + qc) * 3;
            depth[qi] = screen[a + 2] + screen[a + 5] + screen[a + cols * 3 + 2] + screen[a + cols * 3 + 5];
        }
        order.sort(function (x, y) { return depth[y] - depth[x]; });

        var fills = colours[mode];
        ctx.lineWidth = 0.6;                            // the stroke closes hairline seams between quads
        for (var k = 0; k < order.length; k++) {
            var n = order[k];
            var rr = (n / (cols - 1)) | 0, cc = n % (cols - 1);
            var v0 = (rr * cols + cc) * 3, v1 = v0 + 3, v2 = v0 + cols * 3 + 3, v3 = v0 + cols * 3;
            ctx.beginPath();
            ctx.moveTo(screen[v0], screen[v0 + 1]);
            ctx.lineTo(screen[v1], screen[v1 + 1]);
            ctx.lineTo(screen[v2], screen[v2 + 1]);
            ctx.lineTo(screen[v3], screen[v3 + 1]);
            ctx.closePath();
            ctx.fillStyle = fills[n];
            ctx.strokeStyle = fills[n];
            ctx.fill();
            ctx.stroke();
        }

        drawFrame(ctx, view);
    }

    // the floor's outline and the names of the axes
    function drawFrame(ctx, view) {
        var corners = [[-spanX, spanY], [spanX, spanY], [spanX, -spanY], [-spanX, -spanY]]
            .map(function (xy) { return view.project(xy[0], xy[1], BASE); });

        ctx.strokeStyle = view.ink;
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = 1;
        ctx.beginPath();
        corners.forEach(function (p, i) { i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
        ctx.closePath();
        ctx.stroke();

        ctx.globalAlpha = 0.6;
        ctx.fillStyle = view.ink;
        ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        var u = view.project(0, -spanY - 0.18, BASE);
        var v = view.project(spanX + 0.18, 0, BASE);
        ctx.fillText('u', u[0], u[1]);
        ctx.fillText('v', v[0], v[1]);
        ctx.globalAlpha = 1;
    }

    // zoom 1.3: a surface is flatter than a cube, so it can fill more of the canvas
    var plot = Plot3D(canvas, { size: 520, yaw: -0.55, pitch: 0.62, zoom: 1.3, draw: draw });
    plot.redraw();

    modeBox.addEventListener('click', function (e) {
        var btn = e.target.closest('button[data-mode]');
        if (!btn) return;
        mode = btn.dataset.mode;
        modeBox.querySelectorAll('button').forEach(function (b) {
            b.classList.toggle('is-on', b === btn);
        });
        plot.redraw();
    });
})();
