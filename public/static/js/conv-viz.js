/* Convolution visualiser.
 *
 * Ports convolve2d() from app2.py into the browser and animates it: the kernel
 * walks the image pixel by pixel, each output pixel is written as it is
 * computed, and the running k x k neighbourhood is shown alongside.
 *
 * It runs on the file the user has already picked, before anything is
 * submitted, so nothing here talks to the server. The server keeps producing
 * the authoritative /blur result -- the two coming out identical is the point.
 */
(function () {
    'use strict';

    var MAX_DIM = window.MAX_DIM || 500;

    var PANEL = 460;      // the magnified panel is kept at about this many px
    var PANEL_G = 520;    // wide enough that the weights printed on the kernel read
    var BUDGET_MS = 10;   // work per frame, so the page stays at 60fps

    // screen pixels per source pixel in the magnified panel. 0 means "fit",
    // where the scale is whatever shows the entire image at once.
    var ZOOMS = [0, 1, 2, 3, 4, 6, 9, 14, 20, 28];
    var ZOOM_DEFAULT = 7;

    // grid mode: cells on the long side, and how long the kernel dwells on each
    // cell at each speed setting. 0 is "finish it now".
    var GRIDS = [12, 16, 24, 32, 48, 64];
    var CELL_MS = [1400, 800, 420, 200, 90, 0];

    // speed slider position -> output pixels per frame
    var SPEEDS = [1, 4, 25, 200, 2000, Infinity];
    var SPEED_LABELS = ['1 px', '4 px', '25 px', '200 px', '2000 px', 'max'];

    var input = document.getElementById('image');
    var section = document.getElementById('conv');
    if (!input || !section) return;

    var ksize = document.getElementById('ksize');
    var stack = document.getElementById('conv-stack');
    var imgCv = document.getElementById('conv-image');
    var ovCv = document.getElementById('conv-overlay');
    var zoomCv = document.getElementById('conv-zoom');
    var kerCv = document.getElementById('conv-kernel');
    var elK = document.getElementById('conv-k');
    var elPos = document.getElementById('conv-pos');
    var elPct = document.getElementById('conv-pct');
    var elZoomF = document.getElementById('conv-zoomf');   // plate bar, detailed
    var elZoomV = document.getElementById('v-conv-zoom');  // slider readout, short
    var btnPlay = document.getElementById('conv-play');
    var btnStep = document.getElementById('conv-step');
    var btnReset = document.getElementById('conv-reset');
    var speed = document.getElementById('conv-speed');
    var zoomIn = document.getElementById('conv-zoom-level');
    var gridIn = document.getElementById('conv-grid-size');
    var elGridV = document.getElementById('v-conv-grid');
    var elGDims = document.getElementById('conv-gdims');
    var gInCv = document.getElementById('conv-gin');
    var gOutCv = document.getElementById('conv-gout');
    var calcCv = document.getElementById('conv-calc');
    var btnPhoto = document.getElementById('conv-mode-photo');
    var btnGrid = document.getElementById('conv-mode-grid');
    var elSpeed = document.getElementById('v-conv-speed');

    var imgCtx = imgCv.getContext('2d');
    var ovCtx = ovCv.getContext('2d');
    var zoomCtx = zoomCv.getContext('2d');
    var kerCtx = kerCv.getContext('2d');
    var gInCtx = gInCv.getContext('2d');
    var gOutCtx = gOutCv.getContext('2d');
    var calcCtx = calcCv.getContext('2d');

    var dpr = window.devicePixelRatio || 1;
    var ink = getComputedStyle(document.documentElement);
    var COL_INK = ink.getPropertyValue('--ink').trim() || '#000';
    var COL_BG = ink.getPropertyValue('--bg').trim() || '#fff';
    var FONT = '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif';
    var COL_ACC = ink.getPropertyValue('--accent').trim() || '#0a84ff';

    /* ---------- state ---------- */

    var src = null;      // Uint8ClampedArray, RGBA, the unpadded source
    var out = null;      // ImageData being filled in
    var w = 0, h = 0;
    var k = 5, ker = null, half = 2;
    var taps = null;     // k*k*3 scratch for the kernel panel
    var idx = 0;         // next output pixel, row-major
    var playing = false;
    var seekTo = -1;     // when >= 0, run flat out until idx reaches it
    var zx = 0, zy = 0;  // top-left of the magnified window, in source pixels
    var zwW = 0, zwH = 0; // its size, in source pixels
    var zs = 14;         // screen pixels per source pixel inside it
    var scale = 1;       // displayed px per source px on the sweep canvas
    var objectUrl = null;
    var outCv = null;    // offscreen, holds the output alone; the zoom panel samples it
    var outCtx = null;

    // grid mode
    var mode = 'photo';
    var gw = 0, gh = 0;  // grid size in cells
    var gsrc = null;     // RGB input, three values per cell
    var gout = null;     // output, filled in as the kernel visits each cell
    var gtaps = null;    // the k*k pixels under the kernel right now
    var gidx = 0;        // the cell being computed, not yet committed
    var gt = 0;          // 0..1 progress of the move onto that cell
    var lastT = 0;       // for the frame delta grid mode paces itself by

    /* ---------- kernel ---------- */

    // box_kernel(size): every cell 1 / size**2
    function boxKernel(n) {
        var a = new Float64Array(n * n);
        a.fill(1 / (n * n));
        return a;
    }

    // convolve2d() flips the kernel before sliding it. A box kernel is
    // symmetric so this changes nothing here, but the flip is what makes the
    // operation convolution rather than correlation, so it stays.
    function flip(a, n) {
        var f = new Float64Array(n * n);
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) f[i * n + j] = a[(n - 1 - i) * n + (n - 1 - j)];
        return f;
    }

    /* ---------- one output pixel ---------- */

    // out[y,x] = sum_ky sum_kx padded[y+ky, x+kx] * kernel[ky,kx]
    //
    // np.pad(mode="edge") is just index clamping, so no padded copy is built.
    // Alpha is left alone: app2.py only loops over the three colour channels.
    function convPixel(x, y, wantTaps) {
        var r = 0, g = 0, b = 0, t = 0;
        for (var ky = 0; ky < k; ky++) {
            var sy = y + ky - half;
            sy = sy < 0 ? 0 : (sy > h - 1 ? h - 1 : sy);
            var row = sy * w;
            for (var kx = 0; kx < k; kx++, t++) {
                var sx = x + kx - half;
                sx = sx < 0 ? 0 : (sx > w - 1 ? w - 1 : sx);
                var i = (row + sx) << 2;
                var wt = ker[ky * k + kx];
                r += src[i] * wt;
                g += src[i + 1] * wt;
                b += src[i + 2] * wt;
                if (wantTaps) {
                    taps[t * 3] = src[i];
                    taps[t * 3 + 1] = src[i + 1];
                    taps[t * 3 + 2] = src[i + 2];
                }
            }
        }
        return [r, g, b];
    }

    // np.clip(...).astype(np.uint8) truncates, while a Uint8ClampedArray rounds.
    // Truncating here makes the animated result byte-for-byte what /blur returns.
    function write(i, c) {
        var o = i << 2;
        out.data[o] = c[0] | 0;
        out.data[o + 1] = c[1] | 0;
        out.data[o + 2] = c[2] | 0;
        out.data[o + 3] = 255;      // the buffer starts transparent
    }

    /* ---------- loading ---------- */

    function load(file) {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(file);

        var im = new Image();
        im.onload = function () {
            // same cap as img.thumbnail((MAX_DIM, MAX_DIM)) on the server, so
            // this animation and the /blur result are the same pixels
            var s = Math.min(1, MAX_DIM / Math.max(im.naturalWidth, im.naturalHeight));
            w = Math.max(1, Math.round(im.naturalWidth * s));
            h = Math.max(1, Math.round(im.naturalHeight * s));

            imgCv.width = w;
            imgCv.height = h;
            imgCtx.drawImage(im, 0, 0, w, h);
            src = imgCtx.getImageData(0, 0, w, h).data;

            // imgCv keeps the original from here on -- the sweep panel is the
            // "before", so it must not be written over. The output accumulates
            // on its own surface, which the zoom panel shows as the "after".
            outCv = document.createElement('canvas');
            outCv.width = w;
            outCv.height = h;
            outCtx = outCv.getContext('2d');

            applyZoom();

            section.hidden = false;
            sizeOverlay();
            setKernel(parseInt(ksize.value, 10));
        };
        im.src = objectUrl;
    }

    // The panel stays a roughly constant size on screen; the zoom slider sets
    // how many source pixels are packed into it. At the bottom of the range the
    // window is the whole image, which is how you zoom back out.
    function applyZoom() {
        if (!w) return;
        var pick = ZOOMS[zoomIn.value | 0];

        if (pick === 0) {
            zs = Math.min(PANEL / w, PANEL / h);   // fit, so fractional is fine
            zwW = w;
            zwH = h;
        } else {
            zs = pick;
            zwW = Math.min(w, Math.max(1, Math.floor(PANEL / zs)));
            zwH = Math.min(h, Math.max(1, Math.floor(PANEL / zs)));
        }

        var cw = Math.round(zwW * zs), ch = Math.round(zwH * zs);
        zoomCv.style.width = cw + 'px';
        zoomCv.style.height = ch + 'px';
        zoomCv.width = Math.round(cw * dpr);
        zoomCv.height = Math.round(ch * dpr);
        zoomCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

        elZoomV.textContent = pick === 0 ? 'fit' : zs + '×';
        elZoomF.textContent = pick === 0
            ? 'fit · ' + zs.toFixed(2) + ' px per pixel'
            : zs + ' px per pixel · ' + zwW + ' × ' + zwH + ' window';

        // clamp the window back inside the image after a zoom out
        zx = Math.min(zx, Math.max(0, w - zwW));
        zy = Math.min(zy, Math.max(0, h - zwH));
    }

    /* ---------- grid mode ---------- */

    // Reduce the image to something coarse enough to print a number in every
    // cell. The kernel has to fit inside it with room to move, so a large
    // kernel forces a larger grid regardless of the slider.
    function buildGrid() {
        if (!w) return;
        var n = Math.max(GRIDS[gridIn.value | 0], k + 2);

        if (w >= h) { gw = n; gh = Math.max(2, Math.round(n * h / w)); }
        else { gh = n; gw = Math.max(2, Math.round(n * w / h)); }

        // Averaged by hand rather than by drawImage. Shrinking 500px to 20 in one
        // canvas draw is a ~25x reduction, and the built-in filter samples far too
        // few source pixels at that ratio -- the result is aliased and barely
        // resembles the photo. This is an exact area average: every source pixel
        // contributes in proportion to how much of it the cell actually covers,
        // including the fractional slivers at the block edges, which is what makes
        // it agree with a proper box filter at ratios that do not divide evenly.
        gsrc = new Float64Array(gw * gh * 3);
        for (var gy = 0; gy < gh; gy++) {
            var fy0 = gy * h / gh, fy1 = (gy + 1) * h / gh;
            for (var gx = 0; gx < gw; gx++) {
                var fx0 = gx * w / gw, fx1 = (gx + 1) * w / gw;
                var r = 0, g = 0, b = 0, tot = 0;

                for (var yy = Math.floor(fy0); yy < Math.ceil(fy1); yy++) {
                    var cy = Math.min(yy + 1, fy1) - Math.max(yy, fy0);
                    if (cy <= 0) continue;
                    for (var xx = Math.floor(fx0); xx < Math.ceil(fx1); xx++) {
                        var cx = Math.min(xx + 1, fx1) - Math.max(xx, fx0);
                        if (cx <= 0) continue;
                        var a = cy * cx, i = (yy * w + xx) << 2;
                        r += src[i] * a; g += src[i + 1] * a; b += src[i + 2] * a;
                        tot += a;
                    }
                }

                var o = (gy * gw + gx) * 3;
                gsrc[o] = r / tot; gsrc[o + 1] = g / tot; gsrc[o + 2] = b / tot;
            }
        }

        elGridV.textContent = String(n);
        elGDims.textContent = gw + ' × ' + gh + ' cells';
        gridReset();
    }

    // the same sum as convPixel, on the coarse grid
    function convCell(x, y, wantTaps) {
        var r = 0, g = 0, b = 0, t = 0;
        for (var ky = 0; ky < k; ky++) {
            var sy = y + ky - half;
            sy = sy < 0 ? 0 : (sy > gh - 1 ? gh - 1 : sy);
            for (var kx = 0; kx < k; kx++, t++) {
                var sx = x + kx - half;
                sx = sx < 0 ? 0 : (sx > gw - 1 ? gw - 1 : sx);
                var i = (sy * gw + sx) * 3, wt = ker[ky * k + kx];
                r += gsrc[i] * wt;
                g += gsrc[i + 1] * wt;
                b += gsrc[i + 2] * wt;
                if (wantTaps) {
                    gtaps[t * 3] = gsrc[i];
                    gtaps[t * 3 + 1] = gsrc[i + 1];
                    gtaps[t * 3 + 2] = gsrc[i + 2];
                }
            }
        }
        return [r, g, b];
    }

    function commitCell() {
        var c = convCell(gidx % gw, (gidx / gw) | 0, false);
        gout[gidx * 3] = c[0];
        gout[gidx * 3 + 1] = c[1];
        gout[gidx * 3 + 2] = c[2];
        gidx++;
    }

    function gridReset() {
        if (!gsrc) return;
        gout = new Float64Array(gw * gh * 3);        // cells past gidx are simply unfilled
        gidx = 0;
        gt = 0;
    }

    function setKernel(n) {
        k = n;
        half = k >> 1;
        ker = flip(boxKernel(k), k);
        taps = new Float32Array(k * k * 3);
        gtaps = new Float64Array(k * k * 3);
        elK.textContent = k + ' × ' + k;
        buildGrid();          // the grid may have to grow to hold a bigger kernel
        reset();
    }

    function reset() {
        if (!src) return;
        playing = false;
        seekTo = -1;
        idx = 0;
        gridReset();
        // empty, not a copy of the source: the output panel must show only what
        // has actually been computed, or there is nothing to compare against
        out = new ImageData(w, h);
        zx = 0;
        zy = 0;
        btnPlay.textContent = 'Play';
        paint();
    }

    /* ---------- the loop ---------- */

    function tick(now) {
        if (!playing) return;

        var dt = lastT ? Math.min(100, now - lastT) : 16;
        lastT = now;

        if (mode === 'cells') {
            gridTick(dt);
            paint();
            if (playing) requestAnimationFrame(tick);
            else btnPlay.textContent = gidx >= gw * gh ? 'Replay' : 'Play';
            return;
        }

        var t0 = performance.now();
        var per = seekTo >= 0 ? Infinity : SPEEDS[speed.value | 0];
        var done = 0;
        var total = w * h;

        while (idx < total && done < per && performance.now() - t0 < BUDGET_MS) {
            var x = idx % w, y = (idx / w) | 0;
            write(idx, convPixel(x, y, false));
            idx++;
            done++;
            if (seekTo >= 0 && idx >= seekTo) { seekTo = -1; playing = false; break; }
        }

        if (idx >= total) { playing = false; }
        if (!playing) btnPlay.textContent = idx >= total ? 'Replay' : 'Play';

        paint();
        if (playing) requestAnimationFrame(tick);
    }

    // grid mode is paced in milliseconds per cell, and eases between them, so
    // the kernel reads as moving rather than teleporting
    function ease(t) { return t * t * (3 - 2 * t); }

    function gridTick(dt) {
        var total = gw * gh;
        if (gidx >= total) { playing = false; return; }

        var ms = CELL_MS[speed.value | 0];
        if (ms === 0) {
            while (gidx < total) commitCell();
            gt = 0;
            playing = false;
            return;
        }

        gt += dt / ms;
        while (gt >= 1 && gidx < total) { commitCell(); gt -= 1; }
        if (gidx >= total) { gt = 0; playing = false; }
    }

    function play() {
        if (!src) return;
        if (mode === 'cells' ? gidx >= gw * gh : idx >= w * h) reset();
        playing = true;
        lastT = 0;
        btnPlay.textContent = 'Pause';
        requestAnimationFrame(tick);
    }

    function pause() {
        playing = false;
        btnPlay.textContent = 'Play';
    }

    /* ---------- painting ---------- */

    function sizeCanvas(cv, ctx, cw, ch) {
        if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) {
            cv.style.width = cw + 'px';
            cv.style.height = ch + 'px';
            cv.width = Math.round(cw * dpr);
            cv.height = Math.round(ch * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cw, ch);
    }

    /* ---------- grid mode drawing ---------- */

    function paintGrid() {
        if (!gsrc) return;
        var total = gw * gh;
        var cell = Math.max(14, Math.min(56, Math.floor(Math.min(PANEL_G / gw, PANEL_G / gh))));

        // the kernel is between two cells while gt runs 0..1, which is what
        // makes it glide instead of jumping
        var to = Math.min(gidx, total - 1), from = Math.max(0, to - 1);
        var e = ease(gt);
        var bx = (from % gw) + ((to % gw) - (from % gw)) * e;
        var by = ((from / gw) | 0) + (((to / gw) | 0) - ((from / gw) | 0)) * e;
        var cx = Math.round(bx), cy = Math.round(by);

        drawCells(gInCv, gInCtx, gsrc, total, cell, bx, by);
        drawCells(gOutCv, gOutCtx, gout, gidx, cell, -1, -1);
        drawCalc(cx, cy);

        elPos.textContent = 'x ' + cx + ' · y ' + cy;
        elPct.textContent = (gidx / total * 100).toFixed(1) + '%';
    }

    function drawCells(cv, ctx, data, filled, cell, bx, by) {
        sizeCanvas(cv, ctx, gw * cell, gh * cell);

        // the picture stays a picture: every cell keeps its colour, and nothing
        // is written into it. The numbers live on the kernel instead.
        for (var i = 0; i < gw * gh; i++) {
            var x = (i % gw) * cell, y = ((i / gw) | 0) * cell;
            if (i >= filled) continue;                 // not computed yet: leave it empty
            var o = i * 3;
            ctx.fillStyle = 'rgb(' + (data[o] | 0) + ',' + (data[o + 1] | 0) + ',' + (data[o + 2] | 0) + ')';
            ctx.fillRect(x, y, cell, cell);
        }

        // the lattice, over the cells so it reads as one grid
        ctx.globalAlpha = 0.22;
        ctx.strokeStyle = COL_INK;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var c = 0; c <= gw; c++) { ctx.moveTo(c * cell + 0.5, 0); ctx.lineTo(c * cell + 0.5, gh * cell); }
        for (var r = 0; r <= gh; r++) { ctx.moveTo(0, r * cell + 0.5); ctx.lineTo(gw * cell, r * cell + 0.5); }
        ctx.stroke();
        ctx.globalAlpha = 1;

        if (bx < 0) {                                  // output grid: mark where the next value lands
            if (filled < gw * gh) {
                ctx.strokeStyle = COL_ACC;
                ctx.lineWidth = 2;
                ctx.strokeRect((filled % gw) * cell + 1, ((filled / gw) | 0) * cell + 1, cell - 2, cell - 2);
            }
            return;
        }

        drawKernelBox(ctx, bx, by, cell);
    }

    // The kernel as it appears in the classic illustration: a small translucent
    // plate carrying its own weights, riding over the picture. It is drawn whole,
    // so at the borders you can see it hang off -- that overhang is what edge
    // padding fills in.
    function drawKernelBox(ctx, bx, by, cell) {
        var kx = (bx - half) * cell, ky = (by - half) * cell, ks = k * cell;

        ctx.fillStyle = COL_INK;
        ctx.globalAlpha = 0.74;
        ctx.fillRect(kx, ky, ks, ks);
        ctx.globalAlpha = 1;

        var label = '1/' + (k * k);
        if (cell >= 22 && k <= 5) {
            ctx.fillStyle = COL_BG;
            ctx.font = '500 ' + Math.max(8, Math.round(cell * 0.30)) + 'px ' + FONT;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            for (var j = 0; j < k; j++)
                for (var i = 0; i < k; i++)
                    ctx.fillText(label, kx + i * cell + cell / 2, ky + j * cell + cell / 2);

            ctx.globalAlpha = 0.35;                    // the kernel's own cell divisions
            ctx.strokeStyle = COL_BG;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (var g = 1; g < k; g++) {
                ctx.moveTo(kx + g * cell + 0.5, ky); ctx.lineTo(kx + g * cell + 0.5, ky + ks);
                ctx.moveTo(kx, ky + g * cell + 0.5); ctx.lineTo(kx + ks, ky + g * cell + 0.5);
            }
            ctx.stroke();
            ctx.globalAlpha = 1;
        } else {
            // too small to letter every cell, so the weight is stated once above it
            ctx.fillStyle = COL_BG;
            ctx.font = '600 12px ' + FONT;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'alphabetic';
            var ly = ky - 6 < 12 ? ky + ks + 15 : ky - 6;
            ctx.strokeStyle = COL_BG;
            ctx.lineWidth = 3;
            ctx.strokeText(k + '×' + k + ' · each ' + label, Math.max(2, kx), ly);
            ctx.fillStyle = COL_INK;
            ctx.fillText(k + '×' + k + ' · each ' + label, Math.max(2, kx), ly);
        }

        ctx.strokeStyle = COL_ACC;
        ctx.lineWidth = 2;
        ctx.strokeRect(kx, ky, ks, ks);
    }

    // [ the pixels under the kernel ]  × 1/k²  =  [ one output pixel ]
    // For a 3x3 the nine terms are written out per channel underneath, because
    // that is the whole of what a convolution does and it still fits.
    function drawCalc(x, y) {
        var res = convCell(x, y, true);

        var cell = Math.max(12, Math.min(42, Math.floor(160 / k)));
        var grid = cell * k;
        var terms = k === 3;
        var W = PANEL;
        var H = Math.max(grid, 68) + 22 + (terms ? 3 * 18 + 8 : 0);

        sizeCanvas(calcCv, calcCtx, W, H);
        var ctx = calcCtx, mid = grid / 2, ox = 16;

        for (var j = 0; j < k; j++) {
            for (var i = 0; i < k; i++) {
                var t = (j * k + i) * 3;
                ctx.fillStyle = 'rgb(' + (gtaps[t] | 0) + ',' + (gtaps[t + 1] | 0) + ',' + (gtaps[t + 2] | 0) + ')';
                ctx.fillRect(ox + i * cell, j * cell, cell, cell);
            }
        }
        ctx.globalAlpha = 0.22;
        ctx.strokeStyle = COL_INK;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var c = 0; c <= k; c++) {
            ctx.moveTo(ox + c * cell + 0.5, 0); ctx.lineTo(ox + c * cell + 0.5, grid);
            ctx.moveTo(ox, c * cell + 0.5); ctx.lineTo(ox + grid, c * cell + 0.5);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;

        ctx.textBaseline = 'middle';
        ctx.textAlign = 'center';
        ctx.fillStyle = COL_INK;

        var px = ox + grid + 34;
        ctx.font = '400 17px ' + FONT;
        ctx.fillText('×', ox + grid + 15, mid);
        ctx.font = '500 14px ' + FONT;
        ctx.fillText('1/' + (k * k), px, mid - 9);
        ctx.globalAlpha = 0.58;
        ctx.font = '400 12px ' + FONT;
        ctx.fillText('= ' + (1 / (k * k)).toFixed(4), px, mid + 9);
        ctx.globalAlpha = 1;

        var ex = px + 34;
        ctx.font = '400 17px ' + FONT;
        ctx.fillText('=', ex, mid);

        var sw = ex + 20;
        ctx.fillStyle = 'rgb(' + (res[0] | 0) + ',' + (res[1] | 0) + ',' + (res[2] | 0) + ')';
        ctx.fillRect(sw, mid - 22, 44, 44);
        ctx.globalAlpha = 0.22;
        ctx.strokeStyle = COL_INK;
        ctx.strokeRect(sw + 0.5, mid - 21.5, 43, 43);
        ctx.globalAlpha = 1;

        // the three channel sums, which is what actually gets stored
        ctx.fillStyle = COL_INK;
        ctx.textAlign = 'left';
        var names = ['R', 'G', 'B'];
        for (var ch = 0; ch < 3; ch++) {
            ctx.globalAlpha = 0.58;
            ctx.font = '600 11px ' + FONT;
            ctx.fillText(names[ch], sw + 58, mid - 15 + ch * 15);
            ctx.globalAlpha = 1;
            ctx.font = '500 13px ' + FONT;
            ctx.fillText(res[ch].toFixed(1) + '  → ' + (res[ch] | 0), sw + 74, mid - 15 + ch * 15);
        }

        if (!terms) return;

        ctx.globalAlpha = 0.72;
        ctx.font = '400 12px ' + FONT;
        var wt = (1 / (k * k)).toFixed(3);
        for (var c2 = 0; c2 < 3; c2++) {
            var vals = [];
            for (var q = 0; q < k * k; q++) vals.push(Math.round(gtaps[q * 3 + c2]));
            ctx.fillText(names[c2] + '  (' + vals.join(' + ') + ') × ' + wt + '  =  ' + res[c2].toFixed(1),
                ox, grid + 26 + c2 * 18);
        }
        ctx.globalAlpha = 1;
    }

    function paint() {
        if (mode === 'cells') { paintGrid(); return; }
        if (!out) return;
        var total = w * h;
        var at = Math.min(idx, total - 1);
        var x = at % w, y = (at / w) | 0;

        outCtx.putImageData(out, 0, 0);
        recentre(x, y);
        drawOverlay(x, y);
        drawZoom(x, y);
        drawKernel(x, y);

        elPos.textContent = 'x ' + x + ' · y ' + y;
        elPct.textContent = (idx / total * 100).toFixed(1) + '%';
    }

    // jump the window a chunk at a time rather than scrolling it continuously:
    // a still frame is far easier to read than one that drifts every pixel
    function recentre(x, y) {
        var maxX = Math.max(0, w - zwW), maxY = Math.max(0, h - zwH);
        // the margin scales with the window, so a wide view is not re-centred
        // on every single pixel
        var mx = Math.max(2, Math.min(8, zwW >> 3)), my = Math.max(2, Math.min(8, zwH >> 3));
        if (x < zx + mx || x > zx + zwW - 1 - mx)
            zx = Math.min(maxX, Math.max(0, x - (zwW >> 1)));
        if (y < zy + my || y > zy + zwH - 1 - my)
            zy = Math.min(maxY, Math.max(0, y - (zwH >> 1)));
    }

    function sizeOverlay() {
        if (!w) return;
        var box = imgCv.getBoundingClientRect();
        if (!box.width) return;
        scale = box.width / w;
        ovCv.style.width = box.width + 'px';
        ovCv.style.height = box.height + 'px';
        ovCv.width = Math.round(box.width * dpr);
        ovCv.height = Math.round(box.height * dpr);
        ovCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function drawOverlay(x, y) {
        var W = ovCv.width / dpr, H = ovCv.height / dpr;
        ovCtx.clearRect(0, 0, W, H);

        // the magnified window's footprint on the full image
        ovCtx.strokeStyle = COL_INK;
        ovCtx.globalAlpha = 0.30;
        ovCtx.lineWidth = 1;
        if (zwW < w || zwH < h)
            ovCtx.strokeRect(zx * scale, zy * scale, zwW * scale, zwH * scale);

        // the kernel footprint. It is drawn whole, so at the borders you can see
        // it hang off the image -- that overhang is what edge padding fills in.
        ovCtx.globalAlpha = 1;
        ovCtx.lineWidth = 1.5;
        ovCtx.strokeRect((x - half) * scale, (y - half) * scale, k * scale, k * scale);

        // the output pixel being written
        ovCtx.fillStyle = COL_INK;
        ovCtx.fillRect(x * scale, y * scale, Math.max(1, scale), Math.max(1, scale));
    }

    function drawZoom(cx, cy) {
        var W = zwW * zs, H = zwH * zs;

        // a straight crop of the output surface. Sampling it rather than filling
        // per-pixel keeps this cheap at any scale, including the fractional one
        // "fit" uses. Pixels not yet computed are transparent, so they read as
        // empty against the plate rather than as black.
        zoomCtx.imageSmoothingEnabled = false;
        zoomCtx.clearRect(0, 0, W, H);
        zoomCtx.drawImage(outCv, zx, zy, zwW, zwH, 0, 0, W, H);

        // kernel footprint, in window coordinates
        zoomCtx.strokeStyle = COL_INK;
        zoomCtx.lineWidth = zs >= 4 ? 2 : 1;
        zoomCtx.strokeRect((cx - half - zx) * zs, (cy - half - zy) * zs, k * zs, k * zs);

        // the centre tap, which is the pixel being written. Below about 4x a
        // single pixel is too small to outline, so it gets a marker instead.
        var px = (cx - zx) * zs, py = (cy - zy) * zs;
        if (zs >= 4) {
            zoomCtx.lineWidth = 1;
            zoomCtx.strokeStyle = COL_BG;
            zoomCtx.strokeRect(px + 1, py + 1, zs - 2, zs - 2);
            zoomCtx.strokeStyle = COL_INK;
            zoomCtx.strokeRect(px + 0.5, py + 0.5, zs - 1, zs - 1);
        } else {
            zoomCtx.fillStyle = COL_INK;
            zoomCtx.fillRect(px - 1, py - 1, Math.max(2, zs), Math.max(2, zs));
        }
    }

    function drawKernel(x, y) {
        var res = convPixel(x, y, true);

        var cell = Math.min(34, Math.floor(300 / k));
        var grid = cell * k;
        var W = Math.max(grid, 200), H = grid + 78;

        // resizing a canvas resets it, so only do it when the kernel changes
        if (kerCv.width !== W * dpr || kerCv.height !== H * dpr) {
            kerCv.style.width = W + 'px';
            kerCv.style.height = H + 'px';
            kerCv.width = W * dpr;
            kerCv.height = H * dpr;
        }
        kerCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        kerCtx.clearRect(0, 0, W, H);

        var ox = (W - grid) / 2;

        // the k x k neighbourhood the kernel is currently sitting on
        for (var j = 0; j < k; j++) {
            for (var i = 0; i < k; i++) {
                var t = (j * k + i) * 3;
                kerCtx.fillStyle =
                    'rgb(' + (taps[t] | 0) + ',' + (taps[t + 1] | 0) + ',' + (taps[t + 2] | 0) + ')';
                kerCtx.fillRect(ox + i * cell, j * cell, cell, cell);
            }
        }
        kerCtx.strokeStyle = COL_INK;
        kerCtx.globalAlpha = 0.25;
        kerCtx.lineWidth = 1;
        for (var g = 0; g <= k; g++) {
            kerCtx.beginPath();
            kerCtx.moveTo(ox + g * cell + 0.5, 0);
            kerCtx.lineTo(ox + g * cell + 0.5, grid);
            kerCtx.moveTo(ox, g * cell + 0.5);
            kerCtx.lineTo(ox + grid, g * cell + 0.5);
            kerCtx.stroke();
        }
        kerCtx.globalAlpha = 1;

        // every weight in a box kernel is the same, so it is stated once
        kerCtx.fillStyle = COL_INK;
        kerCtx.globalAlpha = 0.58;
        kerCtx.font = '500 12px ' + FONT;
        kerCtx.textAlign = 'center';
        kerCtx.fillText('each × 1/' + (k * k) + '  =  ' + (1 / (k * k)).toFixed(4),
            W / 2, grid + 20);
        kerCtx.fillText('sum', W / 2 - 40, grid + 56);
        kerCtx.globalAlpha = 1;

        // and the pixel that sum produces
        kerCtx.fillStyle = 'rgb(' + Math.round(res[0]) + ',' + Math.round(res[1]) + ',' + Math.round(res[2]) + ')';
        kerCtx.fillRect(W / 2 - 14, grid + 34, 28, 28);
        kerCtx.strokeStyle = COL_INK;
        kerCtx.globalAlpha = 0.25;
        kerCtx.strokeRect(W / 2 - 14.5, grid + 33.5, 29, 29);
        kerCtx.globalAlpha = 1;

        kerCtx.fillStyle = COL_INK;
        kerCtx.textAlign = 'left';
        kerCtx.font = '500 11px ' + FONT;
        kerCtx.fillText(res.map(function (v) { return Math.round(v); }).join(' '),
            W / 2 + 22, grid + 53);
    }

    /* ---------- wiring ---------- */

    input.addEventListener('change', function () {
        var f = input.files[0];
        if (!f) { section.hidden = true; pause(); return; }
        load(f);
    });

    ksize.addEventListener('input', function () {
        // changing k halfway would leave the top of the image blurred by one
        // kernel and the bottom by another, so the run starts over
        if (src) setKernel(parseInt(ksize.value, 10));
    });

    btnPlay.addEventListener('click', function () { playing ? pause() : play(); });
    btnReset.addEventListener('click', reset);

    btnStep.addEventListener('click', function () {
        if (!src) return;
        pause();
        if (mode === 'cells') {
            if (gidx >= gw * gh) return;
            commitCell();
            gt = 0;
        } else {
            if (idx >= w * h) return;
            write(idx, convPixel(idx % w, (idx / w) | 0, false));
            idx++;
        }
        paint();
    });

    function setMode(m) {
        mode = m;
        section.dataset.mode = m;
        btnPhoto.classList.toggle('is-on', m === 'photo');
        btnGrid.classList.toggle('is-on', m === 'cells');
        // the sweep canvas is measured from its rendered box, which was zero
        // while the photo panels were hidden
        if (m === 'photo') sizeOverlay();
        if (src) paint();
    }

    btnPhoto.addEventListener('click', function () { pause(); setMode('photo'); });
    btnGrid.addEventListener('click', function () { pause(); setMode('cells'); });

    gridIn.addEventListener('input', function () {
        if (!src) return;
        pause();
        buildGrid();
        btnPlay.textContent = 'Play';
        paint();
    });

    zoomIn.addEventListener('input', function () {
        if (!src) return;
        applyZoom();
        paint();
    });

    // wheel over the panel walks the same slider, so the two never disagree
    zoomCv.addEventListener('wheel', function (e) {
        if (!src) return;
        e.preventDefault();
        var v = (zoomIn.value | 0) + (e.deltaY < 0 ? 1 : -1);
        v = Math.max(0, Math.min(ZOOMS.length - 1, v));
        if (v === (zoomIn.value | 0)) return;
        zoomIn.value = v;
        applyZoom();
        paint();
    }, { passive: false });

    speed.addEventListener('input', function () {
        elSpeed.textContent = SPEED_LABELS[speed.value | 0] + '/frame';
    });
    elSpeed.textContent = SPEED_LABELS[speed.value | 0] + '/frame';
    zoomIn.value = ZOOM_DEFAULT;   // JS owns the default; the markup only mirrors it
    elGridV.textContent = String(GRIDS[gridIn.value | 0]);
    setMode('photo');

    // click the sweep to jump there: rewind and re-run flat out up to that pixel
    stack.addEventListener('click', function (e) {
        if (!src || mode !== 'photo') return;
        var box = imgCv.getBoundingClientRect();
        var x = Math.floor((e.clientX - box.left) / scale);
        var y = Math.floor((e.clientY - box.top) / scale);
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        var target = y * w + x;
        out = new ImageData(w, h);
        idx = 0;
        seekTo = target;
        playing = true;
        requestAnimationFrame(tick);
    });

    window.addEventListener('resize', function () {
        sizeOverlay();
        if (out) paint();
    });

    // the section is display:none while another operation is selected, so its
    // width is 0 and the overlay cannot be measured until blur is picked again
    if (window.ResizeObserver) {
        new ResizeObserver(function () {
            var before = scale;
            sizeOverlay();
            if (out && scale !== before) paint();
        }).observe(imgCv);
    }
})();
