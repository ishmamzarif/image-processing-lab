/* The three show/hide buttons, each remembered per browser in localStorage.
 *
 *   conv-viz   the blur visualiser             (body[data-viz])
 *   recipe     the Method panel under a result (#recipe-body hidden)
 *   rail       the sidebar                     (body[data-rail])
 *
 * The page reloads on every run, so without remembering, each would reset.
 * Storage calls are wrapped because storage throws outright in a few contexts
 * rather than just coming back empty.
 */

// The visualiser is opt-in.
(function () {
    var btn = document.getElementById('viz-toggle');
    var KEY = 'conv-viz';

    function apply(on, save) {
        document.body.dataset.viz = on ? 'on' : 'off';
        btn.setAttribute('aria-pressed', on ? 'true' : 'false');
        btn.textContent = on ? 'Hide visualisation' : 'Show visualisation';
        if (save) { try { localStorage.setItem(KEY, on ? 'on' : 'off'); } catch (e) { } }
    }

    var saved = 'off';
    try { saved = localStorage.getItem(KEY) || 'off'; } catch (e) { }
    apply(saved === 'on', false);

    btn.addEventListener('click', function () {
        apply(btn.getAttribute('aria-pressed') !== 'true', true);
    });
})();

// The method panel starts collapsed. It only exists on a result page, hence
// the early return.
(function () {
    var btn = document.getElementById('recipe-toggle');
    if (!btn) return;
    var panel = document.getElementById('recipe-body');
    var KEY = 'recipe';

    function apply(open, save) {
        panel.hidden = !open;
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        btn.textContent = open ? 'Hide' : 'Show';
        if (save) { try { localStorage.setItem(KEY, open ? 'on' : 'off'); } catch (e) { } }
    }

    var saved = 'off';
    try { saved = localStorage.getItem(KEY) || 'off'; } catch (e) { }
    apply(saved === 'on', false);

    btn.addEventListener('click', function () { apply(panel.hidden, true); });
})();

// The sidebar can be hidden to give the results the full width. The saved
// choice was already applied by the one-liner at the top of <body> in
// index.html, before first paint; this syncs the button to it and wires it.
(function () {
    var btn = document.getElementById('rail-toggle');
    var KEY = 'rail';

    function apply(open, save) {
        document.body.dataset.rail = open ? 'on' : 'off';
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        btn.setAttribute('aria-label', open ? 'Hide sidebar' : 'Show sidebar');
        btn.title = open ? 'Hide sidebar' : 'Show sidebar';
        if (save) { try { localStorage.setItem(KEY, open ? 'on' : 'off'); } catch (e) { } }
    }

    apply(document.body.dataset.rail !== 'off', false);

    btn.addEventListener('click', function () {
        apply(document.body.dataset.rail === 'off', true);
    });
})();
