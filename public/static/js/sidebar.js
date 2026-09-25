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

    // hold the button while the server works
    form.addEventListener('submit', function (e) {
        var btn = e.submitter;
        if (!btn) return;
        btn.textContent = 'Computing';
        // disabling synchronously cancels the submission in some browsers
        setTimeout(function () {
            var all = form.querySelectorAll('.run button');
            for (var i = 0; i < all.length; i++) all[i].disabled = true;
        }, 0);
    });
})();
