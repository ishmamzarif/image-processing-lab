/* The sidebar form: slider readouts, a few CSS hooks, and the run buttons.
 *
 * Which controls are visible for each operation is pure CSS (see the
 * .rail:has(#op-...:checked) rules in style.css). This file only does what CSS
 * cannot: print each slider's value, and mirror select values onto the form.
 */
(function () {
    var form = document.querySelector('.rail');

    // keep each slider's readout in step with its value
    function readout(id, fmt) {
        var slider = document.getElementById(id);
        var out = document.getElementById('v-' + id);
        function sync() { out.textContent = fmt(slider.value); }
        slider.addEventListener('input', sync);
        sync();
    }

    readout('ksize', function (v) { return v + ' × ' + v; });
    readout('cutoff', function (v) { return v; });
    readout('amount', function (v) { return parseFloat(v).toFixed(1); });
    readout('noise_amount', function (v) { return Math.round(v * 100) + '%'; });
    readout('nwidth', function (v) { return v + ' px'; });
    readout('fcut', function (v) { return v + ' px'; });
    readout('softness', function (v) { return v; });
    readout('keep', function (v) { return v + '%'; });
    readout('angle', function (v) { return v + '°'; });
    readout('gain', function (v) { return parseFloat(v).toFixed(2) + '×'; });

    // offsets show their sign, with a true minus, as on the result page
    function signed(v) { v = parseInt(v, 10); return v > 0 ? '+' + v : v < 0 ? '−' + (-v) : '0'; }
    readout('red', signed);
    readout('green', signed);
    readout('blue', signed);

    // the filters' sliders: amounts as a percentage, then threshold and block size
    function percent(v) { return Math.round(v * 100) + '%'; }
    readout('gray_amount', percent);
    readout('invert_amount', percent);
    readout('vintage_amount', percent);
    readout('threshold', function (v) { return v; });
    readout('block', function (v) { return v + ' px'; });

    // resize: the scale as a percentage
    readout('scale', function (v) { return v + '%'; });

    // CSS cannot see a select's value, so it is mirrored onto the form
    // and the notch / low-pass controls key off that
    var filterSel = document.getElementById('filter');
    function syncFilter() { form.dataset.filter = filterSel.value; }
    filterSel.addEventListener('change', syncFilter);
    syncFilter();

    // the same for the blur kernel, whose angle slider only motion uses
    var kernelSel = document.getElementById('kernel');
    var syncKernel = function () { form.dataset.kernel = kernelSel.value; };
    kernelSel.addEventListener('change', syncKernel);
    syncKernel();

    // and for the filter, which decides which of the five filter sliders shows
    // (data-filter-kind on the form)
    var filterKindSel = document.getElementById('filter_kind');
    var syncFilterKind = function () { form.dataset.filterKind = filterKindSel.value; };
    filterKindSel.addEventListener('change', syncFilterKind);
    syncFilterKind();

    // hold the button while the server works, and show that it is working:
    // body[data-busy] runs the line along the top and dims the canvas (CSS)
    form.addEventListener('submit', function (e) {
        var btn = e.submitter;
        if (!btn) return;
        btn.dataset.label = btn.textContent;
        btn.innerHTML = 'Computing<span class="dots" aria-hidden="true"><i>.</i><i>.</i><i>.</i></span>';
        document.body.dataset.busy = 'on';
        // disabling synchronously cancels the submission in some browsers
        setTimeout(function () {
            var all = form.querySelectorAll('.run button');
            for (var i = 0; i < all.length; i++) all[i].disabled = true;
        }, 0);
    });

    // While a slider is dragged, its value rides above the thumb in a bubble
    // (style.css, "slider bubble"), with the same text as its readout, and
    // the readout gives a small pop on every change. A change from the
    // keyboard shows the bubble for a moment. The bubble is placed from the
    // value alone: the thumb's centre runs from half a thumb in from one end
    // of the track to half a thumb in from the other.
    var THUMB = 14;                                   // px, as in style.css
    var calm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    form.querySelectorAll('input[type="range"]').forEach(function (slider) {
        var out = document.getElementById('v-' + slider.id);
        var field = slider.closest('.field');
        if (!out || !field) return;

        var bubble = document.createElement('span');
        bubble.className = 'bubble';
        bubble.setAttribute('aria-hidden', 'true');
        field.appendChild(bubble);
        var dragging = false, timer = 0;

        function place() {
            var min = parseFloat(slider.min), max = parseFloat(slider.max);
            var frac = max > min ? (parseFloat(slider.value) - min) / (max - min) : 0;
            var x = slider.offsetLeft + THUMB / 2 + frac * (slider.offsetWidth - THUMB);
            bubble.textContent = out.textContent;
            // kept inside the sidebar's padding, pointer still on the thumb
            var half = bubble.offsetWidth / 2, room = 16;
            var left = Math.max(half - room, Math.min(field.offsetWidth + room - half, x));
            bubble.style.left = left + 'px';
            bubble.style.top = slider.offsetTop + 'px';
            bubble.style.setProperty('--nudge', (x - left) + 'px');
        }

        function show(ms) {
            clearTimeout(timer);
            place();
            field.classList.add('is-sliding');
            if (ms) timer = setTimeout(hide, ms);
        }

        function hide() {
            dragging = false;
            field.classList.remove('is-sliding');
        }

        slider.addEventListener('pointerdown', function () { dragging = true; show(0); });
        document.addEventListener('pointerup', function () { if (dragging) hide(); });
        document.addEventListener('pointercancel', function () { if (dragging) hide(); });
        slider.addEventListener('blur', hide);

        // registered after readout()'s own listener, so the text is current
        slider.addEventListener('input', function () {
            if (dragging) place();
            else show(900);
            if (!calm && out.animate) {
                out.animate([{ transform: 'scale(1.2)' }, { transform: 'none' }],
                            { duration: 220, easing: 'cubic-bezier(.2, .7, .2, 1)' });
            }
        });
    });

    // Back and Forward can bring a page back from the browser's cache exactly
    // as it was left, halfway through a run; undo all of the above
    window.addEventListener('pageshow', function (e) {
        if (!e.persisted) return;
        delete document.body.dataset.busy;
        var all = form.querySelectorAll('.run button');
        for (var i = 0; i < all.length; i++) {
            all[i].disabled = false;
            if (all[i].dataset.label) all[i].textContent = all[i].dataset.label;
        }
    });
})();
