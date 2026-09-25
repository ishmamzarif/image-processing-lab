/* Taking the source image with the webcam.
 *
 * "Use camera" in the sidebar opens the live view in the canvas
 * (partials/camera.html). Capture draws the current video frame onto a canvas,
 * saves that as a PNG file, and puts the file into the same file input a
 * picked image goes into. From there everything is as if the photo had been
 * chosen from disk: source-image.js names it, previews it and keeps it across
 * runs, and Apply uploads it.
 *
 * The browser only allows the camera on a secure page, and
 * http://127.0.0.1:5000 counts as one. It asks for permission the first time.
 */
(function () {
    var openBtn = document.getElementById('camera-open');
    var video = document.getElementById('camera-video');
    var status = document.getElementById('camera-status');
    var captureBtn = document.getElementById('camera-capture');
    var cancelBtn = document.getElementById('camera-cancel');
    var errorBox = document.getElementById('camera-error');
    var errorText = document.getElementById('camera-error-text');
    var input = document.getElementById('image');

    var stream = null;   // the live camera feed while the view is open

    // body[data-camera="on"] is what shows the view and hides the rest of the
    // canvas (see "camera" in style.css)
    function showView(on) {
        document.body.dataset.camera = on ? 'on' : 'off';
    }

    function showError(message) {
        errorText.textContent = message;
        errorBox.hidden = false;
        status.textContent = 'not available';
    }

    // turn the camera off: stops the feed, and the camera light goes out
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(function (track) { track.stop(); });
            stream = null;
        }
        video.srcObject = null;
        captureBtn.disabled = true;
    }

    function openCamera() {
        showView(true);
        errorBox.hidden = true;
        status.textContent = 'starting…';
        captureBtn.disabled = true;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('This browser cannot use a camera here. Open the site at http://127.0.0.1:5000.');
            return;
        }

        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function (feed) {
                stream = feed;
                video.srcObject = feed;
                return video.play();
            })
            .then(function () {
                status.textContent = 'live · ' + video.videoWidth + ' × ' + video.videoHeight;
                captureBtn.disabled = false;
            })
            .catch(function (err) {
                stopCamera();
                if (err && err.name === 'NotAllowedError') {
                    showError('Camera permission was refused. Allow it in the browser\'s site settings and try again.');
                } else if (err && err.name === 'NotFoundError') {
                    showError('No camera was found.');
                } else {
                    showError('The camera could not be started' + (err && err.message ? ': ' + err.message : '.'));
                }
            });
    }

    function closeCamera() {
        stopCamera();
        showView(false);
    }

    function capture() {
        var width = video.videoWidth;
        var height = video.videoHeight;
        if (!width || !height) return;

        // draw the current frame, flipped left to right so the photo matches
        // the mirrored view
        var canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        var ctx = canvas.getContext('2d');
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, width, height);

        canvas.toBlob(function (blob) {
            var photo = new File([blob], 'webcam-photo.png', { type: 'image/png' });

            // put the photo into the file input, as if it had been picked
            var transfer = new DataTransfer();
            transfer.items.add(photo);
            input.files = transfer.files;

            // close the view first, so the preview the change brings up is
            // not hidden behind it
            closeCamera();
            input.dispatchEvent(new Event('change'));
        }, 'image/png');
    }

    openBtn.addEventListener('click', openCamera);
    cancelBtn.addEventListener('click', closeCamera);
    captureBtn.addEventListener('click', capture);

    // Esc cancels, as it closes the image viewer
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && document.body.dataset.camera === 'on') closeCamera();
    });
})();
