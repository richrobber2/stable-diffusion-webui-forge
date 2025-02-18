// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

let txt2img_gallery, img2img_gallery, modal = undefined;
let currentPreviewImage = null;
let lastPreviewTime = 0;
const PREVIEW_THROTTLE_MS = 100;  // Throttle preview updates

onAfterUiUpdate(function() {
    if (!txt2img_gallery) {
        txt2img_gallery = attachGalleryListeners("txt2img");
    }
    if (!img2img_gallery) {
        img2img_gallery = attachGalleryListeners("img2img");
    }
    if (!modal) {
        modal = gradioApp().getElementById('lightboxModal');
        modalObserver.observe(modal, {attributes: true, attributeFilter: ['style']});
    }
});

let modalObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutationRecord) {
        let selectedTab = gradioApp().querySelector('#tabs div button.selected')?.innerText;
        if (mutationRecord.target.style.display === 'none' && (selectedTab === 'txt2img' || selectedTab === 'img2img')) {
            gradioApp().getElementById(selectedTab + "_generation_info_button")?.click();
        }
    });
});

function attachGalleryListeners(tab_name) {
    var gallery = gradioApp().querySelector('#' + tab_name + '_gallery');
    gallery?.addEventListener('click', () => gradioApp().getElementById(tab_name + "_generation_info_button").click());
    gallery?.addEventListener('keydown', (e) => {
        if (e.keyCode == 37 || e.keyCode == 39) { // left or right arrow
            gradioApp().getElementById(tab_name + "_generation_info_button").click();
        }
    });
    return gallery;
}

function rememberGallerySelection() {
    currentPreviewImage = selected_gallery_button();
    return true;
}

function getGallerySelectedIndex() {
    if (!currentPreviewImage) return -1;
    return Array.from(all_gallery_buttons()).indexOf(currentPreviewImage);
}

function attachLivePreviewListeners(tabname) {
    const previewContainer = document.createElement('div');
    previewContainer.className = 'live-preview-container';
    previewContainer.style.cssText = 'position:fixed; right:20px; bottom:20px; z-index:10000; background:#333; padding:10px; border-radius:8px;';
    
    const previewCanvas = document.createElement('canvas');
    previewCanvas.className = 'live-preview-image';
    previewCanvas.width = 512;
    previewCanvas.height = 512;
    previewCanvas.style.cssText = 'max-width:512px; max-height:512px;';
    
    const toggleButton = document.createElement('button');
    toggleButton.id = 'live-preview-toggle';
    toggleButton.textContent = 'Live Preview';
    toggleButton.style.cssText = 'display:block; width:100%; margin-top:5px; padding:5px;';
    
    previewContainer.appendChild(previewCanvas);
    previewContainer.appendChild(toggleButton);
    document.body.appendChild(previewContainer);
}
