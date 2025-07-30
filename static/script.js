let canvas = document.getElementById('composite-canvas');
let ctx = canvas.getContext('2d');

let bgImg = new Image();
let p1Img = new Image();
let p2Img = new Image();

let person1 = { x: 50, y: 50, w: 100, h: 100 };
let person2 = { x: 200, y: 50, w: 100, h: 100 };
let scale = 1;
let sessionId = null;

let dragTarget = null;
let resizeTarget = null;
let isDragging = false;
let isResizing = false;
let isAdjusting = false;
let startX, startY, startW, startH;

function previewImage(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

function removeImage(inputId, previewId) {
    document.getElementById(inputId).value = '';
    document.getElementById(previewId).style.display = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawScene() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Scale the background image to fit the canvas while maintaining aspect ratio
    const bgScale = Math.min(canvas.width / bgImg.width, canvas.height / bgImg.height);
    const bgWidth = bgImg.width * bgScale;
    const bgHeight = bgImg.height * bgScale;
    const bgX = (canvas.width - bgWidth) / 2; // Center the background
    const bgY = (canvas.height - bgHeight) / 2; // Center the background
    ctx.drawImage(bgImg, bgX, bgY, bgWidth, bgHeight);
    ctx.drawImage(p1Img, person1.x, person1.y, person1.w, person1.h);
    ctx.drawImage(p2Img, person2.x, person2.y, person2.w, person2.h);
    drawResizeHandle(person1);
    drawResizeHandle(person2);
}

function drawResizeHandle(person) {
    const size = 15;
    ctx.fillStyle = "blue";
    ctx.fillRect(person.x + person.w - size, person.y + person.h - size, size, size);
}

async function uploadImages() {
    const person1File = document.getElementById('person1').files[0];
    const person2File = document.getElementById('person2').files[0];
    const background = document.getElementById('background').files[0];

    if (!person1File || !person2File || !background) {
        alert('Please upload all three images.');
        return;
    }

    document.getElementById('loader').style.display = 'block';

    const formData = new FormData();
    formData.append('person1', person1File);
    formData.append('person2', person2File);
    formData.append('background', background);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            sessionId = result.session_id;
            console.log('Stored session_id:', sessionId);

            person1 = {
                x: result.initial_x1,
                y: result.initial_y1,
                w: result.initial_w1,
                h: result.initial_h1
            };
            person2 = {
                x: result.initial_x2,
                y: result.initial_y2,
                w: result.initial_w2,
                h: result.initial_h2
            };

            document.getElementById('person1-width').value = person1.w;
            document.getElementById('person1-height').value = person1.h;
            document.getElementById('person1-width-value').textContent = person1.w + 'px';
            document.getElementById('person1-height-value').textContent = person1.h + 'px';
            document.getElementById('person1-x').value = person1.x;
            document.getElementById('person1-y').value = person1.y;
            document.getElementById('person1-x-value').textContent = person1.x + 'px';
            document.getElementById('person1-y-value').textContent = person1.y + 'px';
            document.getElementById('person2-width').value = person2.w;
            document.getElementById('person2-height').value = person2.h;
            document.getElementById('person2-width-value').textContent = person2.w + 'px';
            document.getElementById('person2-height-value').textContent = person2.h + 'px';
            document.getElementById('person2-x').value = person2.x;
            document.getElementById('person2-y').value = person2.y;
            document.getElementById('person2-x-value').textContent = person2.x + 'px';
            document.getElementById('person2-y-value').textContent = person2.y + 'px';

            // Calculate scale to maintain aspect ratio and fit within max dimensions (600px width, 450px height)
            const maxWidth = 600;
            const maxHeight = 450;
            const aspectRatio = result.bg_width / result.bg_height;
            let canvasWidth, canvasHeight;

            if (result.bg_width > maxWidth || result.bg_height > maxHeight) {
                if (aspectRatio > maxWidth / maxHeight) {
                    canvasWidth = maxWidth;
                    canvasHeight = maxWidth / aspectRatio;
                } else {
                    canvasHeight = maxHeight;
                    canvasWidth = maxHeight * aspectRatio;
                }
            } else {
                canvasWidth = result.bg_width;
                canvasHeight = result.bg_height;
            }

            scale = canvasWidth / result.bg_width;
            canvas.width = canvasWidth;
            canvas.height = canvasHeight;

            // Apply scale to person coordinates and sizes
            person1.x *= scale;
            person1.y *= scale;
            person1.w *= scale;
            person1.h *= scale;
            person2.x *= scale;
            person2.y *= scale;
            person2.w *= scale;
            person2.h *= scale;

            bgImg.src = result.background_url + '?t=' + new Date().getTime();
            p1Img.src = result.person1_url;
            p2Img.src = result.person2_url;

            await Promise.all([
                new Promise(resolve => bgImg.onload = resolve),
                new Promise(resolve => p1Img.onload = resolve),
                new Promise(resolve => p2Img.onload = resolve)
            ]);
            drawScene();
            document.getElementById('result').style.display = 'block';
            document.querySelectorAll('.p1-controls').forEach(function(el) {
                el.style.display = 'flex';
            });
            document.getElementById('finalize-btn').style.display = 'block';
            document.getElementById('finalize-btn').disabled = false;
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error uploading images:', error);
        alert('Error uploading images: ' + error.message);
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

function updateSize(personId) {
    isAdjusting = true;
    document.getElementById('finalize-btn').disabled = true;
    if (personId === 'person1') {
        person1.w = parseInt(document.getElementById('person1-width').value) || person1.w;
        person1.h = parseInt(document.getElementById('person1-height').value) || person1.h;
        document.getElementById('person1-width-value').textContent = person1.w + 'px';
        document.getElementById('person1-height-value').textContent = person1.h + 'px';
        person1.x = parseInt(document.getElementById('person1-x').value) || person1.x;
        person1.y = parseInt(document.getElementById('person1-y').value) || person1.y;
        document.getElementById('person1-x-value').textContent = person1.x + 'px';
        document.getElementById('person1-y-value').textContent = person1.y + 'px';
    } else if (personId === 'person2') {
        person2.w = parseInt(document.getElementById('person2-width').value) || person2.w;
        person2.h = parseInt(document.getElementById('person2-height').value) || person2.h;
        document.getElementById('person2-width-value').textContent = person2.w + 'px';
        document.getElementById('person2-height-value').textContent = person2.h + 'px';
        person2.x = parseInt(document.getElementById('person2-x').value) || person2.x;
        person2.y = parseInt(document.getElementById('person2-y').value) || person2.y;
        document.getElementById('person2-x-value').textContent = person2.x + 'px';
        document.getElementById('person2-y-value').textContent = person2.y + 'px';
    }
    drawScene();
    isAdjusting = false;
    document.getElementById('finalize-btn').disabled = false;
}

async function saveComposite() {
    try {
        console.log('Saving composite at', new Date().toISOString());
        if (!sessionId) {
            throw new Error('No session ID available. Please upload images first.');
        }
        const response = await fetch('/adjust', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                x1: person1.x / scale,
                y1: person1.y / scale,
                w1: person1.w / scale,
                h1: person1.h / scale,
                x2: person2.x / scale,
                y2: person2.y / scale,
                w2: person2.w / scale,
                h2: person2.h / scale
            })
        });

        const result = await response.json();

        if (response.ok) {
            console.log('Composite saved:', result.composite_url);
            return result.composite_url;
        } else {
            console.error('Error saving composite:', result.error);
            throw new Error('Error saving composite: ' + result.error);
        }
    } catch (error) {
        console.error('Error in saveComposite:', error);
        throw error;
    }
}

async function finalizeImage() {
    if (isAdjusting) {
        console.log('Finalize ignored: adjustment in progress');
        return;
    }

    console.log('finalizeImage triggered at', new Date().toISOString());
    document.getElementById('loader1').style.display = 'block';
    document.getElementById('finalize-btn').disabled = true;
    document.querySelectorAll('.p1-controls').forEach(function(el) {
        el.style.display = 'none';
    });
    
    let statusMessage = document.getElementById('status-message');
    if (!statusMessage) {
        statusMessage = document.createElement('p');
        statusMessage.id = 'status-message';
        document.getElementById('result').appendChild(statusMessage);
    }
    statusMessage.textContent = 'Processing enhanced and harmonized images...';

    try {
        await saveComposite();

        if (!sessionId) {
            throw new Error('No session ID available. Please upload images first.');
        }
        const response = await fetch('/enhance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                x1: person1.x / scale,
                y1: person1.y / scale,
                w1: person1.w / scale,
                h1: person1.h / scale,
                x2: person2.x / scale,
                y2: person2.y / scale,
                w2: person2.w / scale,
                h2: person2.h / scale
            })
        });
        const result = await response.json();
        console.log('Enhance response:', result);

        if (response.ok) {
            if (result.message) {
                statusMessage.textContent = result.message;
            }

            document.getElementById('enhanced-image').src = result.enhanced_url + '?t=' + new Date().getTime();
            document.getElementById('enhanced-image').style.display = 'block';
            document.getElementById('download-enhanced').href = result.enhanced_url;
            document.getElementById('download-enhanced').style.display = 'inline-block';

            if (result.harmonized_url) {
                document.getElementById('harmonized-image').src = result.harmonized_url + '?t=' + new Date().getTime();
                document.getElementById('harmonized-image').style.display = 'block';
                document.getElementById('download-harmonized').href = result.harmonized_url;
                document.getElementById('download-harmonized').style.display = 'inline-block';
            } else {
                document.getElementById('harmonized-image').style.display = 'none';
                document.getElementById('harmonized-image').src = '';
                document.getElementById('download-harmonized').style.display = 'none';
            }
        } else {
            console.error('Enhance failed:', result.error);
            statusMessage.textContent = 'Error: ' + result.error;
            document.getElementById('enhanced-image').style.display = 'none';
            document.getElementById('enhanced-image').src = '';
            document.getElementById('download-enhanced').style.display = 'none';
            document.getElementById('harmonized-image').style.display = 'none';
            document.getElementById('harmonized-image').src = '';
            document.getElementById('download-harmonized').style.display = 'none';
        }
    } catch (error) {
        console.error('Error in finalizeImage:', error);
        statusMessage.textContent = 'Error finalizing image: ' + error.message;
        document.getElementById('enhanced-image').style.display = 'none';
        document.getElementById('enhanced-image').src = '';
        document.getElementById('download-enhanced').style.display = 'none';
        document.getElementById('harmonized-image').style.display = 'none';
        document.getElementById('harmonized-image').src = '';
        document.getElementById('download-harmonized').style.display = 'none';
    } finally {
        document.getElementById('loader1').style.display = 'none';
        document.getElementById('finalize-btn').disabled = false;
    }
}

async function applyHarmonizer() {
    console.log('applyHarmonizer triggered at', new Date().toISOString());
    document.getElementById('loader1').style.display = 'block';
    
    let statusMessage = document.getElementById('status-message');
    if (!statusMessage) {
        statusMessage = document.createElement('p');
        statusMessage.id = 'status-message';
        document.getElementById('result').appendChild(statusMessage);
    }
    statusMessage.textContent = 'Processing harmonized image...';

    try {
        if (!sessionId) {
            throw new Error('No session ID available. Please upload images first.');
        }
        const response = await fetch('/apply_harmonizer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                x1: person1.x / scale,
                y1: person1.y / scale,
                w1: person1.w / scale,
                h1: person1.h / scale,
                x2: person2.x / scale,
                y2: person2.y / scale,
                w2: person2.w / scale,
                h2: person2.h / scale
            })
        });
        const result = await response.json();
        console.log('Harmonizer response:', result);

        if (response.ok) {
            if (result.message) {
                statusMessage.textContent = result.message;
            }
            document.getElementById('harmonized-image').src = result.harmonized_url + '?t=' + new Date().getTime();
            document.getElementById('harmonized-image').style.display = 'block';
            document.getElementById('download-harmonized').href = result.harmonized_url;
            document.getElementById('download-harmonized').style.display = 'inline-block';
        } else {
            console.error('Harmonizer failed:', result.error);
            statusMessage.textContent = 'Error: ' + result.error;
            document.getElementById('harmonized-image').style.display = 'none';
            document.getElementById('harmonized-image').src = '';
            document.getElementById('download-harmonized').style.display = 'none';
        }
    } catch (error) {
        console.error('Error in applyHarmonizer:', error);
        statusMessage.textContent = 'Error applying harmonizer: ' + error.message;
        document.getElementById('harmonized-image').style.display = 'none';
        document.getElementById('harmonized-image').src = '';
        document.getElementById('download-harmonized').style.display = 'none';
    } finally {
        document.getElementById('loader1').style.display = 'none';
    }
}

function inBox(x, y, box) {
    return x >= box.x && x <= box.x + box.w &&
           y >= box.y && y <= box.y + box.h;
}

function inResizeBox(x, y, box) {
    const size = 15;
    return (
        x >= box.x + box.w - size &&
        x <= box.x + box.w &&
        y >= box.y + box.h - size &&
        y <= box.y + box.h
    );
}

canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    if (inResizeBox(mx, my, person1)) {
        resizeTarget = person1;
        isResizing = true;
        isAdjusting = true;
        document.getElementById('finalize-btn').disabled = true;
        startX = mx;
        startY = my;
        startW = person1.w;
        startH = person1.h;
    } else if (inResizeBox(mx, my, person2)) {
        resizeTarget = person2;
        isResizing = true;
        isAdjusting = true;
        document.getElementById('finalize-btn').disabled = true;
        startX = mx;
        startY = my;
        startW = person2.w;
        startH = person2.h;
    } else if (inBox(mx, my, person1)) {
        dragTarget = person1;
        isDragging = true;
        isAdjusting = true;
        document.getElementById('finalize-btn').disabled = true;
        startX = mx;
        startY = my;
    } else if (inBox(mx, my, person2)) {
        dragTarget = person2;
        isDragging = true;
        isAdjusting = true;
        document.getElementById('finalize-btn').disabled = true;
        startX = mx;
        startY = my;
    }
});

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    if (inResizeBox(mx, my, person1) || inResizeBox(mx, my, person2)) {
        canvas.style.cursor = 'nwse-resize';
    } else if (inBox(mx, my, person1) || inBox(mx, my, person2)) {
        canvas.style.cursor = 'move';
    } else {
        canvas.style.cursor = 'default';
    }

    if (isDragging && dragTarget) {
        const dx = mx - startX;
        const dy = my - startY;
        dragTarget.x += dx;
        dragTarget.y += dy;
        if (dragTarget === person1) {
            document.getElementById('person1-x').value = Math.round(dragTarget.x);
            document.getElementById('person1-y').value = Math.round(dragTarget.y);
            document.getElementById('person1-x-value').textContent = Math.round(dragTarget.x) + 'px';
            document.getElementById('person1-y-value').textContent = Math.round(dragTarget.y) + 'px';
        } else if (dragTarget === person2) {
            document.getElementById('person2-x').value = Math.round(dragTarget.x);
            document.getElementById('person2-y').value = Math.round(dragTarget.y);
            document.getElementById('person2-x-value').textContent = Math.round(dragTarget.x) + 'px';
            document.getElementById('person2-y-value').textContent = Math.round(dragTarget.y) + 'px';
        }
        startX = mx;
        startY = my;
        drawScene();
    } else if (isResizing && resizeTarget) {
        const dx = mx - startX;
        const dy = my - startY;

        resizeTarget.w = Math.max(30, startW + dx);
        resizeTarget.h = Math.max(30, startH + dy);

        if (resizeTarget === person1) {
            document.getElementById('person1-width').value = Math.round(resizeTarget.w);
            document.getElementById('person1-height').value = Math.round(resizeTarget.h);
            document.getElementById('person1-width-value').textContent = Math.round(resizeTarget.w) + 'px';
            document.getElementById('person1-height-value').textContent = Math.round(resizeTarget.h) + 'px';
        } else if (resizeTarget === person2) {
            document.getElementById('person2-width').value = Math.round(resizeTarget.w);
            document.getElementById('person2-height').value = Math.round(resizeTarget.h);
            document.getElementById('person2-width-value').textContent = Math.round(resizeTarget.w) + 'px';
            document.getElementById('person2-height-value').textContent = Math.round(resizeTarget.h) + 'px';
        }

        drawScene();
    }
});

canvas.addEventListener('mouseup', () => {
    isDragging = false;
    isResizing = false;
    dragTarget = null;
    resizeTarget = null;
    isAdjusting = false;
    document.getElementById('finalize-btn').disabled = false;
});

canvas.addEventListener('mouseleave', () => {
    isDragging = false;
    isResizing = false;
    dragTarget = null;
    resizeTarget = null;
    isAdjusting = false;
    document.getElementById('finalize-btn').disabled = false;
});

document.getElementById('person1-width').addEventListener('input', () => updateSize('person1'));
document.getElementById('person1-height').addEventListener('input', () => updateSize('person1'));
document.getElementById('person1-x').addEventListener('input', () => updateSize('person1'));
document.getElementById('person1-y').addEventListener('input', () => updateSize('person1'));
document.getElementById('person2-width').addEventListener('input', () => updateSize('person2'));
document.getElementById('person2-height').addEventListener('input', () => updateSize('person2'));
document.getElementById('person2-x').addEventListener('input', () => updateSize('person2'));
document.getElementById('person2-y').addEventListener('input', () => updateSize('person2'));

document.getElementById('person1').addEventListener('change', () => previewImage('person1', 'person1-preview'));
document.getElementById('person2').addEventListener('change', () => previewImage('person2', 'person2-preview'));
document.getElementById('background').addEventListener('change', () => previewImage('background', 'background-preview'));

document.getElementById('upload-btn').addEventListener('click', uploadImages);
document.getElementById('finalize-btn').addEventListener('click', (e) => {
    e.preventDefault();
    finalizeImage();
});

// Cleanup on browser close or navigation away
window.addEventListener('beforeunload', (event) => {
    if (sessionId) {
        fetch(`/cleanup?session_id=${sessionId}`, {
            method: 'DELETE',
            credentials: 'include' // Include if using session cookies
        }).catch(error => console.error('Cleanup request failed:', error));
    }
});
