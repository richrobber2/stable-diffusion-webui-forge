// Utility Classes
class GradioTextAreaBind {
    constructor(elementId, className) {
        this.target = document.querySelector(`#${elementId}.${className} textarea`);
        this.syncLock = false;
        this.previousValue = '';
        this.observer = new MutationObserver(() => {
            if (this.target.value !== this.previousValue) {
                this.previousValue = this.target.value;
                if (!this.syncLock) {
                    this.syncLock = true;
                    callback(this.target.value);
                    this.syncLock = false;
                }
            }
        });
        this.observer.observe(this.target, { characterData: true, subtree: true });
    }

    setValue(newValue) {
        if (this.syncLock) return;

        this.syncLock = true;
        this.target.value = newValue;
        this.previousValue = newValue;

        const inputEvent = new Event('input', { bubbles: true });
        Object.defineProperty(inputEvent, 'target', { value: this.target });
        this.target.dispatchEvent(inputEvent);

        this.syncLock = false;
    }

    listen(callback) {
        // No need for setInterval, handled by MutationObserver
    }
}

// Main Canvas Class
class ForgeCanvas {
    constructor(
        uuid,
        noUpload = false,
        noScribbles = false,
        mask = false,
        initialHeight = 512,
        scribbleColor = '#000000',
        scribbleColorFixed = false,
        scribbleWidth = 4,
        scribbleWidthFixed = false,
        scribbleAlpha = 100,
        scribbleAlphaFixed = false,
        scribbleSoftness = 0,
        scribbleSoftnessFixed = false
    ) {
        this.gradioConfig = typeof gradio_config !== 'undefined' ? gradio_config : null;
        this.uuid = uuid;
        this.noUpload = noUpload;
        this.noScribbles = noScribbles;
        this.mask = mask;
        this.initialHeight = initialHeight;
        this.img = null;
        this.imgX = 0;
        this.imgY = 0;
        this.originalWidth = 0;
        this.originalHeight = 0;
        this.imgScale = 1;
        this.dragging = false;
        this.draggedJustNow = false;
        this.resizing = false;
        this.drawing = false;
        this.scribbleColor = scribbleColor;
        this.scribbleWidth = scribbleWidth;
        this.scribbleAlpha = scribbleAlpha;
        this.scribbleSoftness = scribbleSoftness;
        this.scribbleColorFixed = scribbleColorFixed;
        this.scribbleWidthFixed = scribbleWidthFixed;
        this.scribbleAlphaFixed = scribbleAlphaFixed;
        this.scribbleSoftnessFixed = scribbleSoftnessFixed;
        this.history = [];
        this.historyIndex = -1;
        this.maximized = false;
        this.originalState = {};
        this.contrastPattern = null;
        this.pointerInsideContainer = false;
        this.tempCanvas = document.createElement('canvas');
        this.tempDrawPoints = [];
        this.tempDrawBackground = null;
        this.backgroundGradioBind = new GradioTextAreaBind(this.uuid, 'logical_image_background');
        this.foregroundGradioBind = new GradioTextAreaBind(this.uuid, 'logical_image_foreground');
        this.contrastPatternCanvas = null;
        this.currentMode = 'normal';
        this.currentTool = 'brush';
        this.eraseChanged = false;
        this.start();
    }

    start() {
        const self = this;
        const elements = [
            'imageContainer', 'image', 'resizeLine', 'container', 'toolbar', 'uploadButton',
            'resetButton', 'centerButton', 'removeButton', 'undoButton', 'redoButton',
            'drawingCanvas', 'maxButton', 'minButton', 'scribbleIndicator', 'uploadHint',
            'scribbleColor', 'scribbleColorBlock', 'scribbleWidth', 'widthLabel',
            'scribbleWidthBlock', 'scribbleAlpha', 'alphaLabel', 'scribbleAlphaBlock',
            'scribbleSoftness', 'softnessLabel', 'scribbleSoftnessBlock', 'eraserButton'
        ].reduce((acc, id) => {
            acc[id] = document.getElementById(`${id}_${self.uuid}`);
            return acc;
        }, {});

        const {
            imageContainer, image, resizeLine, container, toolbar, uploadButton,
            resetButton, centerButton, removeButton, undoButton, redoButton,
            drawingCanvas, maxButton, minButton, scribbleIndicator, uploadHint,
            scribbleColor, scribbleColorBlock, scribbleWidth, widthLabel,
            scribbleWidthBlock, scribbleAlpha, alphaLabel, scribbleAlphaBlock,
            scribbleSoftness, softnessLabel, scribbleSoftnessBlock, eraserButton
        } = elements;

        scribbleColor.value = self.scribbleColor;
        scribbleWidth.value = self.scribbleWidth;
        scribbleAlpha.value = self.scribbleAlpha;
        scribbleSoftness.value = self.scribbleSoftness;

        const scribbleIndicatorSize = self.scribbleWidth * 20;
        Object.assign(scribbleIndicator.style, {
            width: `${scribbleIndicatorSize}px`,
            height: `${scribbleIndicatorSize}px`
        });
        container.style.height = `${self.initialHeight}px`;
        drawingCanvas.width = imageContainer.clientWidth;
        drawingCanvas.height = imageContainer.clientHeight;

        const drawingContext = drawingCanvas.getContext('2d');
        self.drawingCanvas = drawingCanvas;

        if (self.noScribbles) {
            ['resetButton', 'undoButton', 'redoButton', 'scribbleColor', 'scribbleColorBlock',
             'scribbleWidthBlock', 'scribbleAlphaBlock', 'scribbleSoftnessBlock', 'scribbleIndicator',
             'drawingCanvas'].forEach(id => elements[id].style.display = 'none');
        }

        if (self.noUpload) {
            uploadButton.style.display = 'none';
            uploadHint.style.display = 'none';
        }

        if (self.mask) {
            ['scribbleColorBlock', 'scribbleAlphaBlock', 'scribbleSoftnessBlock'].forEach(id => elements[id].style.display = 'none');

            const patternCanvas = document.createElement('canvas');
            patternCanvas.width = 20;
            patternCanvas.height = 20;
            const patternContext = patternCanvas.getContext('2d');
            patternContext.fillStyle = '#ffffff';
            patternContext.fillRect(0, 0, 10, 10);
            patternContext.fillRect(10, 10, 10, 10);
            patternContext.fillStyle = '#000000';
            patternContext.fillRect(10, 0, 10, 10);
            patternContext.fillRect(0, 10, 10, 10);
            self.contrastPattern = drawingContext.createPattern(patternCanvas, 'repeat');
            self.contrastPatternCanvas = patternCanvas;
            drawingCanvas.style.opacity = '0.5';
            self.currentMode = 'inpainting';
        }

        if (self.mask || (self.scribbleColorFixed && self.scribbleAlphaFixed && self.scribbleSoftnessFixed)) {
            scribbleSoftnessBlock.style.width = '100%';
            scribbleWidth.style.width = '100%';
            widthLabel.style.display = 'none';
        }

        if (self.scribbleColorFixed) scribbleColorBlock.style.display = 'none';
        if (self.scribbleWidthFixed) scribbleWidthBlock.style.display = 'none';
        if (self.scribbleAlphaFixed) scribbleAlphaBlock.style.display = 'none';
        if (self.scribbleSoftnessFixed) scribbleSoftnessBlock.style.display = 'none';

        const resizeObserver = new ResizeObserver(() => {
            self.adjustInitialPositionAndScale();
            self.drawImage();
        });
        resizeObserver.observe(container);

        const imageInput = document.getElementById(`imageInput_${self.uuid}`);
        imageInput.addEventListener('change', event => self.handleFileUpload(event.target.files[0]));

        uploadButton.addEventListener('click', () => {
            if (!self.noUpload) imageInput.click();
        });

        resetButton.addEventListener('click', () => self.resetImage());
        centerButton.addEventListener('click', () => {
            self.adjustInitialPositionAndScale();
            self.drawImage();
        });
        removeButton.addEventListener('click', () => self.removeImage());
        undoButton.addEventListener('click', () => self.undo());
        redoButton.addEventListener('click', () => self.redo());

        scribbleColor.addEventListener('input', function () {
            self.scribbleColor = this.value;
            scribbleIndicator.style.borderColor = self.scribbleColor;
        });

        scribbleWidth.addEventListener('input', function () {
            self.scribbleWidth = this.value;
            const newSize = self.scribbleWidth * 20;
            Object.assign(scribbleIndicator.style, {
                width: `${newSize}px`,
                height: `${newSize}px`
            });
        });

        scribbleAlpha.addEventListener('input', function () {
            self.scribbleAlpha = this.value;
        });

        scribbleSoftness.addEventListener('input', function () {
            self.scribbleSoftness = this.value;
        });

        eraserButton.addEventListener('click', function () {
            self.currentTool = self.currentTool === 'eraser' ? 'brush' : 'eraser';
            if (self.mask && self.currentTool === 'brush') {
                const context = self.drawingCanvas.getContext('2d');
                context.globalCompositeOperation = 'source-over';
                context.strokeStyle = self.contrastPattern;
            }
            eraserButton.classList.toggle('active');
        });

        drawingCanvas.addEventListener('pointerdown', function (event) {
            if (!self.img || event.button !== 0 || self.noScribbles) return;

            const rect = drawingCanvas.getBoundingClientRect();
            self.drawing = true;
            drawingCanvas.style.cursor = 'crosshair';
            scribbleIndicator.style.display = 'none';
            self.tempDrawPoints = [
                [
                    (event.clientX - rect.left) / self.imgScale,
                    (event.clientY - rect.top) / self.imgScale,
                ],
            ];
            self.tempDrawBackground = drawingContext.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height);

            if (self.currentTool === 'eraser') {
                self.handleErase(event);
            } else {
                if (self.mask) {
                    drawingContext.globalCompositeOperation = 'source-over';
                    drawingContext.strokeStyle = self.contrastPattern;
                }
                self.handleDraw(event);
            }
        });

        drawingCanvas.addEventListener('pointermove', function (event) {
            if (self.drawing) {
                if (self.currentTool === 'eraser') {
                    self.handleErase(event);
                } else {
                    self.handleDraw(event);
                }
            }

            if (self.img && !self.dragging) {
                drawingCanvas.style.cursor = 'crosshair';
            }

            if (self.img && !self.dragging && !self.noScribbles) {
                const containerRect = imageContainer.getBoundingClientRect();
                const indicatorOffset = self.scribbleWidth * 10;
                Object.assign(scribbleIndicator.style, {
                    left: `${event.clientX - containerRect.left - indicatorOffset}px`,
                    top: `${event.clientY - containerRect.top - indicatorOffset}px`,
                    display: 'block'
                });
            }
        });

        drawingCanvas.addEventListener('pointerup', function () {
            if (self.drawing) {
                self.drawing = false;
                self.lastErasePoint = null;
                drawingCanvas.style.cursor = '';

                if (self.eraseChanged) {
                    self.saveState();
                    self.eraseChanged = false;
                }
            }
        });

        drawingCanvas.addEventListener('pointerleave', function () {
            if (self.drawing) {
                self.drawing = false;
                self.lastErasePoint = null;
                drawingCanvas.style.cursor = '';
                scribbleIndicator.style.display = 'none';

                if (self.eraseChanged) {
                    self.saveState();
                    self.eraseChanged = false;
                }
            }
        });

        toolbar.addEventListener('pointerdown', event => event.stopPropagation());

        imageContainer.addEventListener('pointerdown', function (event) {
            const rect = imageContainer.getBoundingClientRect();
            const offsetX = event.clientX - rect.left;
            const offsetY = event.clientY - rect.top;

            if (event.button === 2 && self.isInsideImage(offsetX, offsetY)) {
                self.dragging = true;
                self.offsetX = offsetX - self.imgX;
                self.offsetY = offsetY - self.imgY;
                image.style.cursor = 'grabbing';
                drawingCanvas.style.cursor = 'grabbing';
                scribbleIndicator.style.display = 'none';
            } else if (event.button === 0 && !self.img && !self.noUpload) {
                imageInput.click();
            }
        });

        imageContainer.addEventListener('pointermove', function (event) {
            if (self.dragging) {
                const rect = imageContainer.getBoundingClientRect();
                const mouseX = event.clientX - rect.left;
                const mouseY = event.clientY - rect.top;

                self.imgX = mouseX - self.offsetX;
                self.imgY = mouseY - self.offsetY;
                self.drawImage();
                self.draggedJustNow = true;
            }
        });

        imageContainer.addEventListener('pointerup', event => {
            if (self.dragging) self.handleDragEnd(event, false);
        });

        imageContainer.addEventListener('pointerleave', event => {
            if (self.dragging) self.handleDragEnd(event, true);
        });

        imageContainer.addEventListener('wheel', function (event) {
            if (event.ctrlKey) {
                const brushChange = event.deltaY * -0.01;
                self.scribbleWidth = Math.max(1, self.scribbleWidth + brushChange);
                scribbleWidth.value = self.scribbleWidth;
                const newSize = self.scribbleWidth * 20;
                Object.assign(scribbleIndicator.style, {
                    width: `${newSize}px`,
                    height: `${newSize}px`
                });
                return;
            }

            if (!self.img) return;

            event.preventDefault();

            const rect = imageContainer.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
            const previousScale = self.imgScale;
            const zoomFactor = event.deltaY * -0.001;

            self.imgScale += zoomFactor;
            self.imgScale = Math.max(0.1, self.imgScale);

            const scaleRatio = self.imgScale / previousScale;
            self.imgX = mouseX - (mouseX - self.imgX) * scaleRatio;
            self.imgY = mouseY - (mouseY - self.imgY) * scaleRatio;

            self.drawImage();
        });

        imageContainer.addEventListener('contextmenu', event => {
            event.preventDefault();
            self.draggedJustNow = false;
        });

        imageContainer.addEventListener('pointerover', () => {
            toolbar.style.opacity = '1';
            if (!self.img && !self.noUpload) {
                imageContainer.style.cursor = 'pointer';
            }
        });

        imageContainer.addEventListener('pointerout', () => {
            toolbar.style.opacity = '0';
            image.style.cursor = '';
            drawingCanvas.style.cursor = '';
            imageContainer.style.cursor = '';
            scribbleIndicator.style.display = 'none';
        });

        resizeLine.addEventListener('pointerdown', event => {
            self.resizing = true;
            event.preventDefault();
            event.stopPropagation();
        });

        document.addEventListener('pointermove', event => {
            if (self.resizing) {
                const containerRect = container.getBoundingClientRect();
                const newHeight = event.clientY - containerRect.top;
                container.style.height = `${newHeight}px`;
                event.preventDefault();
                event.stopPropagation();
            }
        });

        document.addEventListener('pointerup', () => self.resizing = false);
        document.addEventListener('pointerleave', () => self.resizing = false);

        ['dragenter', 'dragover'].forEach(eventType => {
            imageContainer.addEventListener(eventType, event => event.preventDefault(), false);
        });

        imageContainer.addEventListener('dragenter', () => {
            image.style.cursor = 'copy';
            drawingCanvas.style.cursor = 'copy';
        });

        imageContainer.addEventListener('dragleave', () => {
            image.style.cursor = '';
            drawingCanvas.style.cursor = '';
        });

        imageContainer.addEventListener('drop', function (event) {
            event.preventDefault();
            const { dataTransfer } = event;
            const { files } = dataTransfer;
            if (files.length > 0) self.handleFileUpload(files[0]);
        });

        imageContainer.addEventListener('pointerenter', () => self.pointerInsideContainer = true);
        imageContainer.addEventListener('pointerleave', () => self.pointerInsideContainer = false);

        document.addEventListener('paste', function (event) {
            if (self.pointerInsideContainer) {
                event.preventDefault();
                event.stopPropagation();
                self.handlePaste(event);
            }
        });

        document.addEventListener('keydown', function (event) {
            if (event.ctrlKey && event.key === 'z') {
                event.preventDefault();
                self.undo();
            } else if (event.ctrlKey && event.key === 'y') {
                event.preventDefault();
                self.redo();
            }
        });

        maxButton.addEventListener('click', () => self.maximize());
        minButton.addEventListener('click', () => self.minimize());

        self.updateUndoRedoButtons();

        self.backgroundGradioBind.listen(base64Data => self.uploadBase64(base64Data));
        self.foregroundGradioBind.listen(base64Data => self.uploadBase64DrawingCanvas(base64Data));

        drawingCanvas.addEventListener('wheel', event => event.preventDefault(), { passive: false });
        drawingCanvas.addEventListener('keydown', event => {
            if (event.ctrlKey || event.metaKey) event.preventDefault();
        });

        drawingCanvas.setAttribute('tabindex', '0');
    }

    handleFileUpload(file) {
        if (file && !this.noUpload) {
            const reader = new FileReader();
            reader.onload = event => this.uploadBase64(event.target.result);
            reader.readAsDataURL(file);
        }
    }

    uploadBase64(base64Data) {
        if (this.gradioConfig && !this.gradioConfig.version.startsWith('4')) return;
        if (!this.gradioConfig) return;

        const img = this.tempImage || new Image();
        img.onload = () => {
            this.img = base64Data;
            this.originalWidth = img.width;
            this.originalHeight = img.height;

            const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            if (drawingCanvas.width !== img.width || drawingCanvas.height !== img.height) {
                drawingCanvas.width = img.width;
                drawingCanvas.height = img.height;
            }

            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.onImageUpload();
            this.saveState();

            document.getElementById(`imageInput_${this.uuid}`).value = null;
            document.getElementById(`uploadHint_${this.uuid}`).style.display = 'none';
        };

        if (base64Data) {
            img.src = base64Data;
        } else {
            this.img = null;
            const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            drawingCanvas.width = 1;
            drawingCanvas.height = 1;
            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.onImageUpload();
            this.saveState();
        }
    }

    uploadBase64DrawingCanvas(base64Data) {
        const img = this.tempImage || new Image();
        img.onload = () => {
            const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            const context = drawingCanvas.getContext('2d');
            context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            context.drawImage(img, 0, 0);
            this.saveState();
        };

        if (base64Data) {
            img.src = base64Data;
        } else {
            const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            const context = drawingCanvas.getContext('2d');
            context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            this.saveState();
        }
    }

    drawImage() {
        const imageElement = document.getElementById(`image_${this.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);

        if (this.img) {
            const scaledWidth = this.originalWidth * this.imgScale;
            const scaledHeight = this.originalHeight * this.imgScale;

            imageElement.src = this.img;
            Object.assign(imageElement.style, {
                width: `${scaledWidth}px`,
                height: `${scaledHeight}px`,
                left: `${this.imgX}px`,
                top: `${this.imgY}px`,
                display: 'block'
            });

            Object.assign(drawingCanvas.style, {
                width: `${scaledWidth}px`,
                height: `${scaledHeight}px`,
                left: `${this.imgX}px`,
                top: `${this.imgY}px`
            });
        } else {
            imageElement.src = '';
            imageElement.style.display = 'none';
        }
    }

    adjustInitialPositionAndScale() {
        const imageContainer = document.getElementById(`imageContainer_${this.uuid}`);
        const containerWidth = imageContainer.clientWidth - 20;
        const containerHeight = imageContainer.clientHeight - 20;

        const scaleX = containerWidth / this.originalWidth;
        const scaleY = containerHeight / this.originalHeight;
        this.imgScale = Math.min(scaleX, scaleY);

        const scaledWidth = this.originalWidth * this.imgScale;
        const scaledHeight = this.originalHeight * this.imgScale;

        this.imgX = (imageContainer.clientWidth - scaledWidth) / 2;
        this.imgY = (imageContainer.clientHeight - scaledHeight) / 2;
    }

    resetImage() {
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const context = drawingCanvas.getContext('2d');
        context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

        this.adjustInitialPositionAndScale();
        this.drawImage();
        this.saveState();
    }

    removeImage() {
        this.img = null;
        const imageElement = document.getElementById(`image_${this.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const context = drawingCanvas.getContext('2d');

        context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        imageElement.src = '';
        imageElement.style.width = '0';
        imageElement.style.height = '0';
        this.saveState();

        if (!this.noUpload) {
            document.getElementById(`uploadHint_${this.uuid}`).style.display = 'block';
        }

        this.onImageUpload();
    }

    isInsideImage(x, y) {
        const scaledWidth = this.originalWidth * this.imgScale;
        const scaledHeight = this.originalHeight * this.imgScale;
        return (
            x > this.imgX &&
            x < this.imgX + scaledWidth &&
            y > this.imgY &&
            y < this.imgY + scaledHeight
        );
    }

    handleDraw(event) {
        const context = this.drawingCanvas.getContext('2d');
        const rect = this.drawingCanvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / this.imgScale;
        const y = (event.clientY - rect.top) / this.imgScale;

        this.tempDrawPoints.push([x, y]);
        context.putImageData(this.tempDrawBackground, 0, 0);
        context.beginPath();
        context.moveTo(this.tempDrawPoints[0][0], this.tempDrawPoints[0][1]);

        for (let i = 1; i < this.tempDrawPoints.length; i++) {
            context.lineTo(this.tempDrawPoints[i][0], this.tempDrawPoints[i][1]);
        }

        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.lineWidth = (this.scribbleWidth / this.imgScale) * 20;

        if (this.mask) {
            context.strokeStyle = this.contrastPattern;
            context.stroke();
            return;
        }

        context.strokeStyle = this.scribbleColor;

        if (this.scribbleAlpha <= 0) {
            context.globalCompositeOperation = 'destination-out';
            context.globalAlpha = 1;
            context.stroke();
            return;
        }

        context.globalCompositeOperation = 'source-over';

        if (this.scribbleSoftness <= 0) {
            context.globalAlpha = this.scribbleAlpha / 100;
            context.stroke();
            return;
        }

        const minLineWidth = context.lineWidth * (1 - this.scribbleSoftness / 150);
        const maxLineWidth = context.lineWidth * (1 + this.scribbleSoftness / 150);
        const steps = Math.round(5 + this.scribbleSoftness / 5);
        const lineWidthIncrement = (maxLineWidth - minLineWidth) / (steps - 1);

        context.globalAlpha = 1 - Math.pow(1 - Math.min(this.scribbleAlpha / 100, 0.95), 1 / steps);

        for (let i = 0; i < steps; i++) {
            context.lineWidth = minLineWidth + lineWidthIncrement * i;
            context.stroke();
        }
    }

    handleErase(event) {
        const context = this.drawingCanvas.getContext('2d');
        const rect = this.drawingCanvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / this.imgScale;
        const y = (event.clientY - rect.top) / this.imgScale;
        const eraserSize = (this.scribbleWidth / this.imgScale) * 20;

        if (!this.lastErasePoint) {
            this.lastErasePoint = { x, y };
        }

        const dx = x - this.lastErasePoint.x;
        const dy = y - this.lastErasePoint.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < eraserSize / 4) {
            context.beginPath();
            context.arc(x, y, eraserSize / 2, 0, Math.PI * 2);
            context.fillStyle = 'rgba(0,0,0,1)';
            context.globalCompositeOperation = 'destination-out';
            context.fill();
            context.closePath();
        } else {
            const steps = Math.ceil(distance / (eraserSize / 4));
            for (let i = 0; i < steps; i++) {
                const t = i / steps;
                const interpX = this.lastErasePoint.x + dx * t;
                const interpY = this.lastErasePoint.y + dy * t;

                context.beginPath();
                context.arc(interpX, interpY, eraserSize / 2, 0, Math.PI * 2);
                context.fillStyle = 'rgba(0,0,0,1)';
                context.globalCompositeOperation = 'destination-out';
                context.fill();
                context.closePath();
            }
        }

        this.lastErasePoint = { x, y };
        this.eraseChanged = true;
    }

    handleDragEnd(event, isLeaving) {
        this.dragging = false;
        const imageElement = document.getElementById(`image_${this.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);

        imageElement.style.cursor = 'grab';
        drawingCanvas.style.cursor = 'grab';
    }

    handlePaste(event) {
        const { items } = event.clipboardData;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.indexOf('image') !== -1) {
                const file = item.getAsFile();
                this.handleFileUpload(file);
                break;
            }
        }
    }

    saveState() {
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const context = drawingCanvas.getContext('2d');
        const imageData = context.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height);

        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push(imageData);
        this.historyIndex++;
        this.updateUndoRedoButtons();
        this.onDrawingCanvasUpload();
    }

    restoreState() {
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const context = drawingCanvas.getContext('2d');
        const imageData = this.history[this.historyIndex];

        context.putImageData(imageData, 0, 0);
        this.onDrawingCanvasUpload();
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.restoreState();
            this.updateUndoRedoButtons();
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.restoreState();
            this.updateUndoRedoButtons();
        }
    }

    updateUndoRedoButtons() {
        const undoButton = document.getElementById(`undoButton_${this.uuid}`);
        const redoButton = document.getElementById(`redoButton_${this.uuid}`);

        undoButton.disabled = this.historyIndex <= 0;
        redoButton.disabled = this.historyIndex >= this.history.length - 1;

        undoButton.style.opacity = undoButton.disabled ? '0.5' : '1';
        redoButton.style.opacity = redoButton.disabled ? '0.5' : '1';
    }

    onImageUpload() {
        if (!this.img) {
            this.backgroundGradioBind.setValue('');
            return;
        }

        const imageElement = document.getElementById(`image_${this.uuid}`);
        const { tempCanvas } = this;
        const context = tempCanvas.getContext('2d');

        tempCanvas.width = this.originalWidth;
        tempCanvas.height = this.originalHeight;
        context.drawImage(imageElement, 0, 0, this.originalWidth, this.originalHeight);

        const base64Data = tempCanvas.toDataURL('image/png');
        this.backgroundGradioBind.setValue(base64Data);
    }

    onDrawingCanvasUpload() {
        if (!this.img) {
            this.foregroundGradioBind.setValue('');
            return;
        }

        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const base64Data = drawingCanvas.toDataURL('image/png');
        this.foregroundGradioBind.setValue(base64Data);
    }

    maximize() {
        if (this.maximized) return;

        const container = document.getElementById(`container_${this.uuid}`);
        const toolbar = document.getElementById(`toolbar_${this.uuid}`);
        const maxButton = document.getElementById(`maxButton_${this.uuid}`);
        const minButton = document.getElementById(`minButton_${this.uuid}`);

        this.originalState = {
            width: container.style.width,
            height: container.style.height,
            top: container.style.top,
            left: container.style.left,
            position: container.style.position,
            zIndex: container.style.zIndex,
        };

        Object.assign(container.style, {
            width: '100vw',
            height: '100vh',
            top: '0',
            left: '0',
            position: 'fixed',
            zIndex: '1000'
        });

        maxButton.style.display = 'none';
        minButton.style.display = 'inline-block';
        this.maximized = true;
    }

    minimize() {
        if (!this.maximized) return;

        const container = document.getElementById(`container_${this.uuid}`);
        const maxButton = document.getElementById(`maxButton_${this.uuid}`);
        const minButton = document.getElementById(`minButton_${this.uuid}`);

        Object.assign(container.style, {
            width: this.originalState.width,
            height: this.originalState.height,
            top: this.originalState.top,
            left: this.originalState.left,
            position: this.originalState.position,
            zIndex: this.originalState.zIndex
        });

        maxButton.style.display = 'inline-block';
        minButton.style.display = 'none';
        this.maximized = false;
    }
}

// Constants
const True = true;
const False = false;
