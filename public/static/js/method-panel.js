/* The Method panel under a result: syntax colours for the code, typeset maths. */

// Restrained Python highlighting for the .code blocks. It rewrites text nodes
// only, so markup already in a block (the call tree's <b> and <i>) stays where
// it is. Identifiers are matched as a whole so the digit in a name like fft2 is
// not mistaken for a number.
(function () {
    var KEYWORDS = /^(and|as|break|continue|def|elif|else|for|from|if|import|in|is|lambda|not|or|pass|return|while|with|None|True|False)$/;
    var TOKEN = /(#[^\n]*)|("[^"\n]*"|'[^'\n]*')|\b(\d+\.?\d*(?:e-?\d+)?)|([A-Za-z_]\w*)/g;

    function paint(node) {
        var text = node.nodeValue, frag = document.createDocumentFragment();
        var last = 0, m;
        TOKEN.lastIndex = 0;
        while ((m = TOKEN.exec(text))) {
            var cls = m[1] ? 'tok-c' : m[2] ? 'tok-s' : m[3] ? 'tok-n'
                    : KEYWORDS.test(m[4]) ? 'tok-k' : null;
            if (!cls) continue;
            frag.appendChild(document.createTextNode(text.slice(last, m.index)));
            var span = document.createElement('span');
            span.className = cls;
            span.textContent = m[0];
            frag.appendChild(span);
            last = TOKEN.lastIndex;
        }
        if (!last) return;
        frag.appendChild(document.createTextNode(text.slice(last)));
        node.parentNode.replaceChild(frag, node);
    }

    // .no-hl opts out: the kernel's weights are all numbers, and painting
    // every one of them in the accent would drown the table
    document.querySelectorAll('.code:not(.no-hl)').forEach(function (block) {
        var walk = document.createTreeWalker(block, NodeFilter.SHOW_TEXT), nodes = [];
        while (walk.nextNode()) nodes.push(walk.currentNode);
        nodes.forEach(paint);
    });
})();

// Typeset the maths once KaTeX (loaded with defer in index.html, only on result
// pages) has run. fleqn keeps each equation flush left, in line with its code
// block. Offline, KaTeX is missing and the raw TeX simply stays visible.
document.addEventListener('DOMContentLoaded', function () {
    if (!window.katex) return;
    document.querySelectorAll('.tex').forEach(function (el) {
        katex.render(el.textContent, el, { displayMode: true, fleqn: true, throwOnError: false });
        el.classList.add('is-typeset');
    });
});
