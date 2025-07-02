// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() {

}

function getGallerySelectedIndex() {

}

function request(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? '0' + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}


var originalAppTitle = undefined;

onUiLoaded(function() {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    var title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}


function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    var dateStart = new Date();
    var wasEverActive = false;
    var parentProgressbar = progressbarContainer.parentNode;
    var wakeLock = null;

    var requestWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock) return;
        try {
            wakeLock = await navigator.wakeLock.request('screen');
        } catch (err) {
            console.error('Wake Lock is not supported.');
        }
    };

    var releaseWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error('Wake Lock release failed', err);
        }
    };

    var divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    var divInner = document.createElement('div');
    divInner.className = 'progress';

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    var livePreview = null;

    var removeProgressBar = function() {
        releaseWakeLock();
        if (!divProgress) return;

        setTitle("");
        parentProgressbar.removeChild(divProgress);
        if (gallery && livePreview) gallery.removeChild(livePreview);
        atEnd();

        divProgress = null;
    };

    // --- WebSocket-based progress and live preview updates ---
    var wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var wsUrl = wsProtocol + '//' + location.host + '/ws/progress/' + encodeURIComponent(id_task);
    var ws = new WebSocket(wsUrl);
    var progressTimeout = null;

    ws.onopen = function() {
        requestWakeLock();
    };
    ws.onmessage = function(event) {
        var res = JSON.parse(event.data);
        let progressText = "";
        divInner.style.width = ((res.progress || 0) * 100.0) + '%';
        divInner.style.background = res.progress ? "" : "transparent";
        if (res.progress > 0) {
            progressText = ((res.progress || 0) * 100.0).toFixed(0) + '%';
        }
        setTitle(progressText);
        divInner.textContent = progressText;
        var elapsedFromStart = (new Date() - dateStart) / 1000;
        if (res.active) wasEverActive = true;
        if (!res.active && wasEverActive) {
            ws.close();
            removeProgressBar();
            return;
        }
        if (elapsedFromStart > inactivityTimeout && !res.queued && !res.active) {
            ws.close();
            removeProgressBar();
            return;
        }
        if (onProgress) {
            onProgress(res);
        }
        // --- Live preview update ---
        if (res.live_preview && gallery) {
            var img = new Image();
            img.onload = function() {
                if (!livePreview) {
                    livePreview = document.createElement('div');
                    livePreview.className = 'livePreview';
                    gallery.insertBefore(livePreview, gallery.firstElementChild);
                }
                livePreview.appendChild(img);
                if (livePreview.childElementCount > 2) {
                    livePreview.removeChild(livePreview.firstElementChild);
                }
            };
            img.src = res.live_preview;
        }
    };
    ws.onerror = function() {
        ws.close();
        removeProgressBar();
    };
    ws.onclose = function() {
        removeProgressBar();
    };
    // No polling for live preview needed anymore
}
