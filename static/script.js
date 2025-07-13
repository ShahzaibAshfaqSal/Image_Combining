let canvas = document.getElementById('composite-canvas');
let ctx = canvas.getContext('2d');

let bgImg = new Image();
let p1Img = new Image();
let p2Img = new Image();

let person1 = { x: 50, y: 50, w: 100, h: 100 };
let person2 = { x: 200, y: 50, w: 100, h: 100 };
let scale = 1;

let dragTarget = null;
let resizeTarget = null;
let isDragging = false;
let isResizing = false;
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
    ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
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
            document.getElementById('person2-width').value = person2.w;
            document.getElementById('person2-height').value = person2.h;
            document.getElementById('person2-width-value').textContent = person2.w + 'px';
            document.getElementById('person2-height-value').textContent = person2.h + 'px';

            scale = Math.min(1000 / result.bg_width, 600 / result.bg_height, 1);
            canvas.width = result.bg_width * scale;
            canvas.height = result.bg_height * scale;

            person1.x *= scale;
            person1.y *= scale;
            person1.w *= scale;
            person1.h *= scale;
            person2.x *= scale;
            person2.y *= scale;
            person2.w *= scale;
            person2.h *= scale;

            drawScene();
            bgImg.src = result.background_url + '?t=' + new Date().getTime();
            p1Img.src = result.person1_url;
            p2Img.src = result.person2_url;

            document.getElementById('result').style.display = 'block';
            document.getElementById('finalize-btn').style.display = 'block';
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error uploading images: ' + error.message);
    } finally {
        document.getElementById('loader').style.display = 'none';
    }
}

async function updateSize(personId) {
    if (personId === 'person1') {
        person1.w = parseInt(document.getElementById('person1-width').value);
        person1.h = parseInt(document.getElementById('person1-height').value);
        document.getElementById('person1-width-value').textContent = person1.w + 'px';
        document.getElementById('person1-height-value').textContent = person1.h + 'px';
    } else if (personId === 'person2') {
        person2.w = parseInt(document.getElementById('person2-width').value);
        person2.h = parseInt(document.getElementById('person2-height').value);
        document.getElementById('person2-width-value').textContent = person2.w + 'px';
        document.getElementById('person2-height-value').textContent = person2.h + 'px';
    }

    await updateComposite();
    drawScene();
}

async function updateComposite() {
    try {
        const response = await fetch('/adjust', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
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
            bgImg.onload = drawScene;
        } else {
            alert('Error updating composite: ' + result.error);
        }
    } catch (error) {
        alert('Error updating composite: ' + error.message);
    }
}

async function finalizeImage() {
    document.getElementById('loader1').style.display = 'block';
    document.getElementById('finalize-btn').disabled = true;
    try {
        // Option to enhance whole image or just faces
        const enhanceWhole = confirm("Enhance whole image with GFPGAN? (Cancel for face crops only)");
        const response = await fetch('/enhance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                x1: person1.x / scale,
                y1: person1.y / scale,
                w1: person1.w / scale,
                h1: person1.h / scale,
                x2: person2.x / scale,
                y2: person2.y / scale,
                w2: person2.w / scale,
                h2: person2.h / scale,
                enhance_whole: enhanceWhole
            })
        });
        const result = await response.json();

        if (response.ok) {
            document.getElementById('enhanced-image').src = result.enhanced_url + '?t=' + new Date().getTime();
            document.getElementById('enhanced-image').style.display = 'block';
            document.getElementById('download-enhanced').href = result.enhanced_url;
            document.getElementById('download-enhanced').style.display = 'inline-block';
        
            document.getElementById('loader1').style.display = 'none';
            
            // Prompt for harmonization and handle response
            if (result.message && confirm(result.message)) {
                await applyHarmonizer();
            } else {
                console.log('Harmonization skipped by user.');
            }
        } else {
            alert('Error finalizing image: ' + result.error);
        }
    } catch (error) {
        console.error('Error in finalizeImage:', error);
        alert('Error finalizing image: ' + error.message);
    } finally {
        
        document.getElementById('finalize-btn').disabled = false;
    }
}

async function applyHarmonizer() {
    document.getElementById('loader2').style.display = 'block';
    try {
        const response = await fetch('/apply_harmonizer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
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
            document.getElementById('harmonized-image').src = result.harmonized_url + '?t=' + new Date().getTime();
            document.getElementById('harmonized-image').style.display = 'block';
            document.getElementById('download-harmonized').href = result.harmonized_url;
            document.getElementById('download-harmonized').style.display = 'inline-block';
        } else {
            alert('Error applying harmonizer: ' + result.error);
            document.getElementById('harmonized-image').style.display = 'none';
            document.getElementById('harmonized-image').src = '';
            document.getElementById('download-harmonized').style.display = 'none';
        }
    } catch (error) {
        console.error('Error in applyHarmonizer:', error);
        alert('Error applying harmonizer: ' + error.message);
        document.getElementById('harmonized-image').style.display = 'none';
        document.getElementById('harmonized-image').src = '';
        document.getElementById('download-harmonized').style.display = 'none';
    } finally {
        document.getElementById('loader2').style.display = 'none';
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
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (inResizeBox(mx, my, person1)) {
        resizeTarget = person1;
        isResizing = true;
        startX = mx;
        startY = my;
        startW = person1.w;
        startH = person1.h;
    } else if (inResizeBox(mx, my, person2)) {
        resizeTarget = person2;
        isResizing = true;
        startX = mx;
        startY = my;
        startW = person2.w;
        startH = person2.h;
    } else if (inBox(mx, my, person1)) {
        dragTarget = person1;
        isDragging = true;
        startX = mx;
        startY = my;
    } else if (inBox(mx, my, person2)) {
        dragTarget = person2;
        isDragging = true;
        startX = mx;
        startY = my;
    }
});

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

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
    if (isDragging || isResizing) {
        updateComposite();
    }
    isDragging = false;
    isResizing = false;
    dragTarget = null;
    resizeTarget = null;
});

canvas.addEventListener('mouseleave', () => {
    if (isDragging || isResizing) {
        updateComposite();
    }
    isDragging = false;
    isResizing = false;
    dragTarget = null;
    resizeTarget = null;
});
