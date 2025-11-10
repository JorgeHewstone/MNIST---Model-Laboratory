// Wait until the full HTML is loaded
document.addEventListener("DOMContentLoaded", () => {

    // --- DOM References (Drawing Panel) ---
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('result');
    const modelInfoDiv = document.getElementById('model-info');

    // --- DOM References (Parameters Panel) ---
    const modelTypeSelect = document.getElementById('model_type');
    const optimizerSelect = document.getElementById('optimizer');
    const batchSizeSelect = document.getElementById('batch_size');
    const lrRangeSelect = document.getElementById('lr_range');
    const epochsInput = document.getElementById('epochs');
    const numLayersInput = document.getElementById('num_layers');
    const layersContainer = document.getElementById('layers-container');
    const useDropoutCheckbox = document.getElementById('use_dropout');

    // --- Canvas Logic ---
    let drawing = false;
    ctx.lineWidth = 25;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    function clearCanvas() {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultDiv.textContent = '...';
        modelInfoDiv.innerHTML = '<p>Draw a digit to predict.</p>';
    }

    function startDraw(e) {
        drawing = true;
        draw(e);
    }

    function endDraw() {
        drawing = false;
        ctx.beginPath();
    }

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        // Adjust for scaling if the display is zoomed
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    function draw(e) {
        if (!drawing) return;
        const pos = getMousePos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }
    
    // Start with a blank canvas
    clearCanvas();

    // --- Helpers (Parameters & URLs) ---
    function getHiddenLayersConfig() {
        const layerInputs = document.querySelectorAll('.layer-size');
        return Array.from(layerInputs).map(input => parseInt(input.value));
    }

    function buildTrainingURL() {
        // Build a link to the training module with current params as query string.
        const params = new URLSearchParams();
        params.set('model_type', modelTypeSelect.value);
        params.set('optimizer', optimizerSelect.value);
        params.set('batch_size', String(parseInt(batchSizeSelect.value)));
        params.set('lr_range', lrRangeSelect.value); // "Low", "Medium Low", "Medium High", "High"
        params.set('epochs', String(parseInt(epochsInput.value)));
        params.set('use_dropout', String(!!(useDropoutCheckbox && useDropoutCheckbox.checked)));
        const layers = getHiddenLayersConfig();
        if (layers.length > 0) {
            params.set('layers', layers.join(',')); // e.g., "128,64,32"
        }

        // You can adjust the base if your training app is elsewhere
        return `http://127.0.0.1:8000/?${params.toString()}`;
    }

    function renderTrainCta(messageText = 'Model not trained yet.') {
        const trainUrl = buildTrainingURL();
        resultDiv.textContent = 'N/A';
        modelInfoDiv.innerHTML = `
            <div style="text-align:left">
                <p style="color:#dc3545;margin-bottom:10px;">${messageText}</p>
                <a href="${trainUrl}" 
                   style="
                        display:inline-block;
                        padding:10px 12px;
                        border-radius:8px;
                        text-decoration:none;
                        font-weight:600;
                        border:1px solid #d0d7de;
                        background:#ffffff;
                        color:#0f172a;
                        margin-right:8px;"
                   title="Go to the training module with these parameters">
                   Train this configuration
                </a>
                <a href="${trainUrl}" target="_blank" rel="noopener"
                   style="font-size:0.9em; color:#007bff; text-decoration:none;">
                   Open in new tab
                </a>
            </div>
        `;
    }

    // --- Form Logic (Parameters) ---
    function updateLayerInputs() {
        const count = parseInt(numLayersInput.value) || 0;
        layersContainer.innerHTML = '';
        const defaults = [128, 64, 32, 16];
        for (let i = 0; i < count; i++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'layer-size';
            input.placeholder = `Layer ${i + 1}`;
            input.value = defaults[i] || 64;
            input.required = true;
            layersContainer.appendChild(input);
        }
    }

    function setLayerValues(valuesArray) {
        if (!numLayersInput || !layersContainer) return;
        numLayersInput.value = valuesArray.length;
        updateLayerInputs();
        const inputs = layersContainer.querySelectorAll('.layer-size');
        inputs.forEach((inp, idx) => {
            if (idx < valuesArray.length) inp.value = parseInt(valuesArray[idx]);
        });
    }

    function parseQueryParams() {
        const params = new URLSearchParams(window.location.search);
        const getStr = (k) => {
            const v = params.get(k);
            return (v !== null && v !== '') ? v : undefined;
        };
        const getInt = (k) => {
            const v = params.get(k);
            const n = parseInt(v);
            return (!isNaN(n)) ? n : undefined;
        };
        const q = {};
        q.model_type  = getStr('model_type');
        q.optimizer   = getStr('optimizer');
        q.batch_size  = getInt('batch_size');
        q.lr_range    = getStr('lr_range');     // "Low" | "Medium Low" | ...
        q.epochs      = getInt('epochs');
        q.use_dropout = (getStr('use_dropout') === 'true');
        const layers  = getStr('layers');
        if (layers) {
            q.layers = layers.split(',').map(s => parseInt(s.trim())).filter(x => !isNaN(x));
        }
        return q;
    }

    function prefillFromQuery() {
        const q = parseQueryParams();
        if (!q) return;

        if (q.model_type && modelTypeSelect) modelTypeSelect.value = q.model_type;
        if (q.optimizer && optimizerSelect) optimizerSelect.value = q.optimizer;
        if (Number.isInteger(q.batch_size) && batchSizeSelect) batchSizeSelect.value = String(q.batch_size);
        if (q.lr_range && lrRangeSelect) lrRangeSelect.value = q.lr_range;
        if (Number.isInteger(q.epochs) && epochsInput) epochsInput.value = String(q.epochs);
        if (typeof q.use_dropout === 'boolean' && useDropoutCheckbox) useDropoutCheckbox.checked = q.use_dropout;

        if (Array.isArray(q.layers) && q.layers.length > 0) {
            setLayerValues(q.layers);
        } else {
            updateLayerInputs();
        }
    }


    // --- Prediction Logic ---
    async function predict() {
        resultDiv.textContent = 'Searching...';
        modelInfoDiv.innerHTML = '';
        
        // 1) Build a 28x28 image for the API
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        tempCanvas.getContext('2d').drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);
        const imageData = tempCanvas.toDataURL('image/png');
        
        // 2) Collect *all* model parameters
        const hidden_layers_config = getHiddenLayersConfig();

        const modelParams = {
            model_type: modelTypeSelect.value,
            optimizer: optimizerSelect.value,
            learning_rate_range: lrRangeSelect.value, // "Low", "Medium Low", "Medium High", "High"
            batch_size: parseInt(batchSizeSelect.value),
            epochs: parseInt(epochsInput.value),
            hidden_layers_config: hidden_layers_config,
            use_dropout: !!(useDropoutCheckbox && useDropoutCheckbox.checked)

        };

        // 3) Send to API
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    params: modelParams,
                    image_data: imageData
                })
            });

            const data = await response.json();

            if (data.error) {
                // "Model not trained yet" is an expected outcome
                renderTrainCta(data.error);
            } else {
                // Success
                resultDiv.textContent = `Prediction: ${data.prediction} (${data.confidence})`;
                modelInfoDiv.innerHTML = `
                    <p>
                        <strong>Matched model:</strong> ${data.found_model.id}<br>
                        <strong>Accuracy:</strong> ${data.found_model.accuracy}<br>
                        <strong>LR (actual):</strong> ${data.found_model.learning_rate}<br>
                        <strong>Epochs:</strong> ${data.found_model.epochs}
                    </p>
                `;
            }
        } catch (err) {
            resultDiv.textContent = 'Error';
            modelInfoDiv.innerHTML = '<p style="color: red;">API connection error.</p>';
            console.error(err);
        }
    }

    // --- Event Listeners ---
    // Canvas
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', endDraw); // Stop if mouse leaves

    // Controls
    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', predict);

    // Parameters form
    numLayersInput.addEventListener('input', updateLayerInputs);

    // --- Initialization ---
    updateLayerInputs(); // Generate layer inputs on load
    prefillFromQuery();
});
