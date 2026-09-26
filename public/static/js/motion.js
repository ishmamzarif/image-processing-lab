/* The sliding highlight behind the segmented buttons (.conv__mode): Photo /
 * Grid on the visualiser, Before / After / Both on the colour cube, and the
 * modes of every 3D view.
 *
 * The scripts that own each group only move the .is-on class between its
 * buttons. This adds one block of ink per group (.conv__pill) and slides it
 * under whichever button has the class, so none of those scripts has to know
 * about it. A group inside a hidden section measures as nothing, so the block
 * is placed again whenever the group's size changes, which includes the
 * moment it is shown.
 *
 * The slide itself is a CSS transition (style.css, "motion"), switched on only
 * after the first real placement, so the block does not fly in from the left
 * when the page loads or a hidden group first appears.
 */
(function () {
    'use strict';

    document.querySelectorAll('.conv__mode').forEach(function (group) {
        var pill = document.createElement('span');
        pill.className = 'conv__pill';
        pill.setAttribute('aria-hidden', 'true');
        group.insertBefore(pill, group.firstChild);
        group.classList.add('has-pill');

        var ready = false;

        function place() {
            var on = group.querySelector('button.is-on');
            if (!on || !group.offsetWidth) return;
            pill.style.width = on.offsetWidth + 'px';
            pill.style.transform = 'translateX(' + on.offsetLeft + 'px)';
            if (ready) return;
            // a frame after the first placement, so it is not itself animated
            ready = true;
            requestAnimationFrame(function () {
                requestAnimationFrame(function () { group.classList.add('is-ready'); });
            });
        }

        new MutationObserver(place).observe(group, { subtree: true, attributes: true, attributeFilter: ['class'] });
        if (window.ResizeObserver) new ResizeObserver(place).observe(group);
        place();
    });
})();
