/* A very small 3D view on a 2D canvas, shared by the colour cube (rgb-cube.js)
 * and the surfaces (surface-3d.js).
 *
 * The scene lives in world coordinates inside [-1, 1] on every axis, with z
 * pointing up. The camera is orthographic: it turns round the vertical axis
 * (yaw) and tilts down towards the scene (pitch), both changed by dragging.
 * Double-click puts the view back where it started.
 *
 * options: size (the widest the canvas gets, in CSS px), yaw and pitch (the
 * starting view, in radians), zoom (1 leaves room for a whole cube turned any
 * way; a flatter scene can afford more) and draw.
 *
 * Nothing here knows what is being drawn. The caller hands over draw(ctx, view)
 * and calls redraw() whenever its data changes; draw uses view.project(x, y, z)
 * to get [screen x, screen y, depth], where a larger depth is further away, so
 * sorting by depth, largest first, paints back to front.
 */
(function () {
    'use strict';

    var css = getComputedStyle(document.documentElement);

    // the page's own colours, so the plots follow the design tokens in style.css
    function colour(name, fallback) {
        return css.getPropertyValue(name).trim() || fallback;
    }

    window.Plot3D = function (canvas, options) {
        var ctx = canvas.getContext('2d');
        var dpr = window.devicePixelRatio || 1;
        var yaw0 = options.yaw, pitch0 = options.pitch;
        var yaw = yaw0, pitch = pitch0;
        var width = 0, height = 0, scale = 1;
        var queued = false;
        var cosY, sinY, cosP, sinP;

        var view = {
            ink: colour('--ink', '#000'),
            bg: colour('--bg', '#fff'),
            accent: colour('--accent', '#0a84ff'),

            // world -> [screen x, screen y, depth]. After the yaw, the camera
            // looks along +y1 from above at angle `pitch`, so a point further
            // back (larger y1) sits higher on screen, and a higher one (larger
            // z) is nearer.
            project: function (x, y, z) {
                var x1 = x * cosY - y * sinY;
                var y1 = x * sinY + y * cosY;
                return [
                    width / 2 + scale * x1,
                    height / 2 - scale * (z * cosP + y1 * sinP),
                    y1 * cosP - z * sinP
                ];
            }
        };

        // The canvas is square-ish and as wide as the row allows, up to
        // options.size. It is measured on the .pair, since a plate hugs its
        // content and would only report the canvas's own width back; 30 px is
        // the well's padding and border. The scene's corners reach sqrt(3) from
        // the centre when turned, so the scale leaves that much room.
        function resize() {
            var row = canvas.closest('.pair');
            var room = row ? row.clientWidth - 30 : options.size;
            width = Math.max(220, Math.min(options.size, room || options.size));
            height = Math.round(width * 0.86);
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            scale = Math.min(width, height) * 0.29 * (options.zoom || 1);
        }

        function paint() {
            queued = false;
            if (!canvas.offsetParent) return;          // hidden: nothing to see
            if (!width || canvas.width !== Math.round(width * dpr)) resize();
            cosY = Math.cos(yaw); sinY = Math.sin(yaw);
            cosP = Math.cos(pitch); sinP = Math.sin(pitch);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, width, height);
            options.draw(ctx, view);
        }

        // many events can arrive per frame, and the scene is drawn once
        function redraw() {
            if (queued) return;
            queued = true;
            requestAnimationFrame(paint);
        }

        // --- turning the view by dragging ---
        var dragging = false, lastX = 0, lastY = 0;

        canvas.style.cursor = 'grab';
        canvas.style.touchAction = 'none';            // a drag turns the plot, not the page

        canvas.addEventListener('pointerdown', function (e) {
            dragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            canvas.setPointerCapture(e.pointerId);
            canvas.style.cursor = 'grabbing';
        });

        canvas.addEventListener('pointermove', function (e) {
            if (!dragging) return;
            yaw -= (e.clientX - lastX) * 0.01;
            // from level with the floor to straight down, and no further
            pitch = Math.max(0, Math.min(Math.PI / 2, pitch + (e.clientY - lastY) * 0.01));
            lastX = e.clientX;
            lastY = e.clientY;
            redraw();
        });

        function release() {
            dragging = false;
            canvas.style.cursor = 'grab';
        }
        canvas.addEventListener('pointerup', release);
        canvas.addEventListener('pointercancel', release);

        canvas.addEventListener('dblclick', function () {
            yaw = yaw0;
            pitch = pitch0;
            redraw();
        });

        window.addEventListener('resize', function () {
            width = 0;
            redraw();
        });

        return { redraw: redraw };
    };
})();
