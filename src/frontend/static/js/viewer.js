/* Full-screen viewer: click any image plate to see it large, one at a time. */
(function () {
    var viewer = document.getElementById('viewer');
    var vImg = document.getElementById('viewer-img');
    var vCap = document.getElementById('viewer-cap');

    // magnify by the largest whole-number factor that still fits, so every
    // source pixel becomes an exact square block instead of being resampled
    function magnify(nw, nh) {
        if (!nw || !nh) return '';
        var f = Math.min(window.innerWidth * 0.92 / nw,
                         window.innerHeight * 0.86 / nh);
        f = Math.max(1, Math.floor(f));
        vImg.style.width = (nw * f) + 'px';
        vImg.style.height = (nh * f) + 'px';
        return f;
    }

    document.querySelectorAll('.plate__well').forEach(function (well) {
        well.addEventListener('click', function () {
            var img = well.querySelector('img');
            if (!img.getAttribute('src')) return;
            vImg.src = img.src;
            var f = magnify(img.naturalWidth, img.naturalHeight);
            vCap.textContent = well.dataset.cap +
                (f > 1 ? ' — ' + f + '×' : '') + ' — click or esc to close';
            viewer.classList.add('is-open');
        });
    });

    // keep the download link from also opening the viewer
    document.querySelectorAll('.save').forEach(function (a) {
        a.addEventListener('click', function (e) { e.stopPropagation(); });
    });

    function close() {
        viewer.classList.remove('is-open');
        vImg.removeAttribute('src');
    }

    viewer.addEventListener('click', close);
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') close();
    });
})();
