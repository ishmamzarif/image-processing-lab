/* Height fields you can turn: every 3D view except the colour cube.
 *
 * Each one is a .surface3d block (partials/surface_3d.html) holding a canvas,
 * a row of mode buttons and its data as JSON. The data comes from the server,
 * already pooled and scaled to 0..1; backend/common/surface.py describes the
 * format. In short: a list of modes, each a grid of heights, optionally with
 * a tint (drawn in the accent colour) or the picture's own colours.
 *
 * The flat plates show a spectrum as brightness, where a noise peak is one
 * pale dot among many. As height it is a spike standing out of a low field,
 * and a filter is a shape you can see: the notch a hole, the high-pass a bowl.
 * The same drawing turns a picture into terrain, its brightness as height.
 *
 * The modes of one view can have different grids (a resized patch has more or
 * fewer pixels than the original); each is stretched over the same floor.
 */
(function () {
    'use strict';

    if (!window.Plot3D) return;

    var BASE = -0.5;       // world height of the floor

    // "#0a84ff" -> [10, 132, 255]
    function rgbOf(hex, fallback) {
        var m = /^#([0-9a-f]{6})$/i.exec(hex);
        if (!m) return fallback;
        var n = parseInt(m[1], 16);
        return [n >> 16, (n >> 8) & 255, n & 255];
    }

    // One mode's geometry: where its vertices sit on the floor, and scratch
    // space for projecting and sorting them.
    function prepare(mode) {
        var rows = mode.z.length, cols = mode.z[0].length;
        var quads = (rows - 1) * (cols - 1);
        var order = [];
        for (var q = 0; q < quads; q++) order.push(q);
        return {
            z: mode.z, tint: mode.tint, rgb: mode.rgb, plain: mode.plain,
            rows: rows, cols: cols,
            screen: new Float64Array(rows * cols * 3),
            depth: new Float64Array(quads),
            order: order,
            fills: null
        };
    }

    function setUp(canvas, data, modeBox) {
        var HEIGHT = data.height || 1.05;     // world height of a 1
        var axes = data.axes || ['u', 'v'];
        var modes = data.modes.map(prepare);
        var current = 0;

        // The longer side of the floor spans -1..1 and the shorter keeps the
        // grid's shape. Taken from the first mode, so every mode shares it.
        var r0 = modes[0].rows, c0 = modes[0].cols;
        var spanX = c0 >= r0 ? 1 : c0 / r0;
        var spanY = r0 >= c0 ? 1 : r0 / c0;

        // Each quad's colour depends on the data and the light, not on the
        // view, so it is worked out once per mode rather than on every frame.
        // A light from the upper left shades the slopes so the shape reads.
        // Grey darkens with height, or stays one light grey for a plain mode;
        // a tinted quad takes the accent, and a picture keeps its own colours.
        function shade(m, accent) {
            var z = m.z, rows = m.rows, cols = m.cols;
            var dx = 2 * spanX / Math.max(1, cols - 1);
            var dy = 2 * spanY / Math.max(1, rows - 1);
            var lx = -0.45, ly = 0.5, lz = 0.74;              // towards the light, unit length
            var rgb = m.rgb && m.rgb.map(function (row) {
                return row.map(function (hex) { return rgbOf(hex, [128, 128, 128]); });
            });
            var out = [];
            for (var r = 0; r < rows - 1; r++) {
                for (var c = 0; c < cols - 1; c++) {
                    var h = (z[r][c] + z[r][c + 1] + z[r + 1][c] + z[r + 1][c + 1]) / 4;

                    // the slope across the quad, and its normal (-dz/dx, -dz/dy, 1)
                    var sx = ((z[r][c + 1] - z[r][c]) + (z[r + 1][c + 1] - z[r + 1][c])) / 2 * HEIGHT / dx;
                    var sy = -((z[r + 1][c] - z[r][c]) + (z[r + 1][c + 1] - z[r][c + 1])) / 2 * HEIGHT / dy;
                    var len = Math.sqrt(sx * sx + sy * sy + 1);
                    var light = Math.max(0, (-sx * lx - sy * ly + lz) / len);
                    var s = 0.6 + 0.4 * light;

                    var tinted = m.tint && (m.tint[r][c] || m.tint[r][c + 1] || m.tint[r + 1][c] || m.tint[r + 1][c + 1]);
                    var col;
                    if (tinted) {
                        col = accent;
                    } else if (rgb) {
                        col = [0, 1, 2].map(function (k) {
                            return (rgb[r][c][k] + rgb[r][c + 1][k] + rgb[r + 1][c][k] + rgb[r + 1][c + 1][k]) / 4;
                        });
                    } else {
                        var g = m.plain ? 205 : 228 - 178 * h;
                        col = [g, g, g];
                    }
                    out.push('rgb(' + col.map(function (v) { return Math.round(v * s); }).join(',') + ')');
                }
            }
            return out;
        }

        function draw(ctx, view) {
            var m = modes[current];
            if (!m.fills) m.fills = shade(m, rgbOf(view.accent, [10, 132, 255]));
            var z = m.z, rows = m.rows, cols = m.cols, screen = m.screen, depth = m.depth;

            // every vertex once
            for (var r = 0; r < rows; r++) {
                for (var c = 0; c < cols; c++) {
                    var wx = (c / Math.max(1, cols - 1) * 2 - 1) * spanX;
                    var wy = (1 - r / Math.max(1, rows - 1) * 2) * spanY;     // row 0 at the back
                    var p = view.project(wx, wy, BASE + z[r][c] * HEIGHT);
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
            m.order.sort(function (x, y) { return depth[y] - depth[x]; });

            ctx.lineWidth = 0.6;                            // the stroke closes hairline seams between quads
            for (var k = 0; k < m.order.length; k++) {
                var n = m.order[k];
                var rr = (n / (cols - 1)) | 0, cc = n % (cols - 1);
                var v0 = (rr * cols + cc) * 3, v1 = v0 + 3, v2 = v0 + cols * 3 + 3, v3 = v0 + cols * 3;
                ctx.beginPath();
                ctx.moveTo(screen[v0], screen[v0 + 1]);
                ctx.lineTo(screen[v1], screen[v1 + 1]);
                ctx.lineTo(screen[v2], screen[v2 + 1]);
                ctx.lineTo(screen[v3], screen[v3 + 1]);
                ctx.closePath();
                ctx.fillStyle = m.fills[n];
                ctx.strokeStyle = m.fills[n];
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
            ctx.fillText(axes[0], u[0], u[1]);
            ctx.fillText(axes[1], v[0], v[1]);
            ctx.globalAlpha = 1;
        }

        // zoom 1.3: a surface is flatter than a cube, so it can fill more of the canvas
        var plot = Plot3D(canvas, { size: 520, yaw: -0.55, pitch: 0.62, zoom: data.zoom || 1.3, draw: draw });
        plot.redraw();

        if (!modeBox) return;
        modeBox.addEventListener('click', function (e) {
            var btn = e.target.closest('button[data-mode]');
            if (!btn) return;
            current = +btn.dataset.mode;
            modeBox.querySelectorAll('button').forEach(function (b) {
                b.classList.toggle('is-on', b === btn);
            });
            plot.redraw();
        });
    }

    document.querySelectorAll('.surface3d').forEach(function (box) {
        var canvas = box.querySelector('canvas');
        var source = box.querySelector('script[type="application/json"]');
        if (canvas && source) setUp(canvas, JSON.parse(source.textContent), box.querySelector('.conv__mode'));
    });
})();
