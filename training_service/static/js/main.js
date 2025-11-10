// Wait for the entire HTML document to be loaded before running the script
document.addEventListener("DOMContentLoaded", () => {
    
    // --- DOM References ---
    const numLayersInput = document.getElementById('num_layers');
    const layersContainer = document.getElementById('layers-container');
    const form = document.getElementById('train-form');
    const submitBtn = document.getElementById('submit-btn');
    const logArea = document.getElementById('log-area');
    const finalResultDiv = document.getElementById('final-result');
    const chartCtx = document.getElementById('loss-chart').getContext('2d');

    // Form fields to (optionally) prefill
    const modelTypeSelect = document.getElementById('model_type');
    const optimizerSelect = document.getElementById('optimizer');
    const numEpochsInput = document.getElementById('num_epochs');
    const batchSizeInput = document.getElementById('batch_size');
    const lrRangeSelect = document.getElementById('lr_range'); 
    const useDropoutCheckbox = document.getElementById('use_dropout');
    const abortBtn = document.getElementById('abort-btn');

    // --- Global Variables ---
    let lossChart;
    let websocket;
    let lossHistory = []; // For the moving average

    // --- LR ranges mapping (range -> [lo, hi]) ---
    const LR_RANGES = {
        "Low":         [1e-5, 1e-4],
        "Medium Low":  [1e-4, 5e-4],
        "Medium High": [5e-4, 1e-3],
        "High":        [1e-3, 1e-2]
    };

    function lrFromRange(rangeName) {
        const rng = LR_RANGES[rangeName];
        if (!rng) return 1e-3; // fallback
        return (rng[0] + rng[1]) / 2.0; // midpoint
    }

    // --- Initialize Chart ---
    function initChart() {
        if (lossChart) {
            lossChart.destroy();
        }
        lossHistory = []; // Clear history for the moving average
        
        lossChart = new Chart(chartCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Raw Loss',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 0.2)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        borderWidth: 1,
                        pointRadius: 0
                    },
                    {
                        label: 'Moving Average (Trend)',
                        data: [],
                        borderColor: 'rgba(235, 99, 132, 1)',
                        backgroundColor: 'rgba(235, 99, 132, 0.2)',
                        borderWidth: 2,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Steps' } },
                    y: { type: 'logarithmic', title: { display: true, text: 'Loss' } }
                },
                animation: false
            }
        });
    }

    // --- Function to add to the log ---
    function addLog(message) {
        logArea.textContent += message + '\n';
        logArea.scrollTop = logArea.scrollHeight;
    }

    // --- Function to update the chart ---
    function updateChart(step, loss) {
        lossHistory.push(loss);
        let windowSize = 25;
        windowSize = Math.min(windowSize, lossHistory.length);

        let movingAverage = 0;
        if (lossHistory.length > 0) {
            const recentHistory = lossHistory.slice(-windowSize);
            const sum = recentHistory.reduce((a, b) => a + b, 0);
            movingAverage = sum / recentHistory.length;
        }

        lossChart.data.datasets[0].data.push({ x: step, y: loss });
        if (movingAverage > 0) {
            lossChart.data.datasets[1].data.push({ x: step, y: movingAverage });
        }

        lossChart.update('none');
    }

    // --- Helpers: layers, query params, prefill ---
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
        numLayersInput.value = valuesArray.length;
        updateLayerInputs();
        const inputs = layersContainer.querySelectorAll('.layer-size');
        inputs.forEach((inp, idx) => {
            if (idx < valuesArray.length) inp.value = parseInt(valuesArray[idx]);
        });
    }

    function parseQueryParams() {
        const params = new URLSearchParams(window.location.search);
        const result = {};
        const getStr = (k) => {
            const v = params.get(k);
            return (v !== null && v !== '') ? v : undefined;
        };
        const getInt = (k) => {
            const v = params.get(k);
            return (v !== null && v !== '' && !Number.isNaN(parseInt(v))) ? parseInt(v) : undefined;
        };

        result.model_type = getStr('model_type');
        result.optimizer  = getStr('optimizer');
        result.batch_size = getInt('batch_size');
        result.lr_range   = getStr('lr_range');  // we keep using range names
        result.num_epochs = getInt('epochs');
        result.use_dropout = params.get('use_dropout') === 'true';
        const layers      = getStr('layers');    // CSV
        if (layers) {
            result.layers = layers.split(',').map(x => parseInt(x.trim())).filter(x => !Number.isNaN(x));
        }
        return result;
    }

    function prefillFromQuery() {
        const q = parseQueryParams();
        if (!q) return;

        if (q.model_type && modelTypeSelect) modelTypeSelect.value = q.model_type;
        if (q.optimizer && optimizerSelect) optimizerSelect.value = q.optimizer;
        if (Number.isInteger(q.batch_size) && batchSizeInput) batchSizeInput.value = String(q.batch_size);
        if (Number.isInteger(q.num_epochs) && numEpochsInput) numEpochsInput.value = String(q.num_epochs);

        if (q.lr_range && LR_RANGES[q.lr_range] && lrRangeSelect) {
            lrRangeSelect.value = q.lr_range;
        }

        if (q.layers && Array.isArray(q.layers) && q.layers.length > 0) {
            setLayerValues(q.layers);
        }

        if (typeof q.use_dropout === 'boolean' && useDropoutCheckbox) {
            useDropoutCheckbox.checked = q.use_dropout;
            }
    }

    function buildTestURL({ modelType, optimizer, batchSize, lrRangeName, epochs, hiddenLayers }) {
        const params = new URLSearchParams();
        params.set('model_type', modelType);
        params.set('optimizer', optimizer);
        params.set('batch_size', String(batchSize));
        params.set('lr_range', lrRangeName);         
        params.set('epochs', String(epochs));
        if (hiddenLayers && hiddenLayers.length > 0) {
            params.set('layers', hiddenLayers.join(','));
        }
        if (typeof use_dropout === 'boolean') params.set('use_dropout', String(use_dropout));
        return `http://127.0.0.1:8001/?${params.toString()}`;
        
    }

    // --- Form submission (WebSocket training) ---
    function handleTrain(event) {
        event.preventDefault();
        submitBtn.disabled = true;
        submitBtn.textContent = 'Training...';
        abortBtn.disabled = false;
        abortBtn.textContent = 'Abort Training';
        abortBtn.onclick = () => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                addLog('Abort requested by user…');
                websocket.send(JSON.stringify({ type: 'abort' }));
                abortBtn.disabled = true;
                abortBtn.textContent = 'Aborting…';
            }
        };


        // Clear previous UI
        initChart();
        logArea.textContent = '';
        finalResultDiv.innerHTML = '';
        
        // 1) Collect config
        const layerInputs = document.querySelectorAll('.layer-size');
        const hidden_layers_config = Array.from(layerInputs).map(input => parseInt(input.value));

        const selectedRange = lrRangeSelect.value;          // keep the *name* for test URL
        const learning_rate = lrFromRange(selectedRange);   // numeric midpoint for training
        const use_dropout = !!useDropoutCheckbox.checked;   
        const cfg = {
            model_type: modelTypeSelect.value,
            optimizer: optimizerSelect.value,
            learning_rate, // numeric midpoint
            num_epochs: parseInt(numEpochsInput.value),
            batch_size: parseInt(batchSizeInput.value),
            num_layers: parseInt(numLayersInput.value),
            hidden_layers_config,
            use_dropout  
        };

        // Prepare test URL now (so we don't depend on server echo)
        const testUrl = buildTestURL({
            modelType: cfg.model_type,
            optimizer: cfg.optimizer,
            batchSize: cfg.batch_size,
            lrRangeName: selectedRange,
            epochs: cfg.num_epochs,
            hiddenLayers: hidden_layers_config,
            use_dropout 
        });

        // 2) Connect WebSocket
        const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const ws_url = `${wsProtocol}://${window.location.host}/ws/train`;
        websocket = new WebSocket(ws_url);

        websocket.onopen = () => {
            addLog('Connected to server...');
            addLog(`Using LR midpoint from "${selectedRange}": ${learning_rate.toExponential(2)}`);
            addLog('Sending configuration...');
            websocket.send(JSON.stringify(cfg));
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);

            switch (data.type) {
                case 'log':
                    addLog(data.message);
                    break;

                case 'exists': {
                    const msg = data.message || 'Model already exists. Overwrite?';
                    const details = data.existing ? 
                        `\n\nExisting ID: ${data.existing.id}\nAccuracy: ${data.existing.accuracy}\nLR: ${data.existing.learning_rate}\nEpochs: ${data.existing.num_epochs}\nBatch: ${data.existing.batch_size}\nDropout: ${data.existing.use_dropout}\nLayers: ${data.existing.hidden_layers_config}` 
                        : '';
                    const ok = window.confirm(msg + details);

                    websocket.send(JSON.stringify({ type: 'overwrite', value: ok }));

                    if (!ok) {
                        addLog('User declined overwrite. Training will be skipped.');
                    } else {
                        addLog('User accepted overwrite. Removing previous model…');
                    }
                    break;
                }

                case 'loss':
                    updateChart(data.step, data.loss);
                    break;

                case 'result': {
                    // Re-enable button even if skipped or success
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Start Training';
                    abortBtn.disabled = true;
                    abortBtn.textContent = 'Abort Training';

                    if (data.skipped) {
                        finalResultDiv.innerHTML = `<h3>Skipped</h3><p>${data.message || 'Training skipped.'}</p>`;
                        return;
                    }

                    addLog('Training Complete!');
                    if (lossChart) lossChart.update();

                    // Collect current UI config to craft test URL (mirrors what we trained)
                    const layerInputs = document.querySelectorAll('.layer-size');
                    const hidden_layers_config = Array.from(layerInputs).map(inp => parseInt(inp.value));
                    const modelType = document.getElementById('model_type').value;
                    const optimizer = document.getElementById('optimizer').value;
                    const batchSize = parseInt(document.getElementById('batch_size').value);
                    const epochs    = parseInt(document.getElementById('num_epochs').value);
                    const lrRangeName = lrRangeSelect ? lrRangeSelect.value : 'High';
                    const use_dropout = !!(useDropoutCheckbox && useDropoutCheckbox.checked);

                    const testUrl = buildTestURL({
                        modelType, optimizer, batchSize, lrRangeName, epochs,
                        hiddenLayers: hidden_layers_config,
                        use_dropout
                    });

                    finalResultDiv.innerHTML = `
                        <h3>Final Results!</h3>
                        <p><strong>Model:</strong> ${data.model_type}</p>
                        <p><strong>Optimizer:</strong> ${data.optimizer}</p>
                        <p><strong>Final Accuracy:</strong> ${data.accuracy.toFixed(2)}%</p>
                        <p><strong>Total Time:</strong> ${data.training_time.toFixed(2)}s</p>
                        <div style="margin-top:12px;">
                            <a href="${testUrl}"
                            style="display:inline-block;padding:10px 12px;border-radius:8px;text-decoration:none;font-weight:600;border:1px solid #d0d7de;background:#ffffff;color:#0f172a;margin-right:8px;">
                            Test this model
                            </a>
                            <a href="${testUrl}" target="_blank" rel="noopener"
                            style="font-size:0.9em; color:#007bff; text-decoration:none;">
                            Open in new tab
                            </a>
                        </div>
                    `;
                    break;
                }

                case 'error':
                    addLog(`ERROR: ${data.message}`);
                    finalResultDiv.innerHTML = `<h3 style="color: red;">Error</h3><p>${data.message}</p>`;
                    // Re-enable button on error
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Start Training';
                    abortBtn.disabled = true;
                    abortBtn.textContent = 'Abort Training';
                    break;

                case 'aborted':
                    addLog('Training aborted by user.');
                    finalResultDiv.innerHTML = `<h3>Aborted</h3><p>${data.message || 'Training aborted.'}</p>`;
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Start Training';
                    abortBtn.disabled = true;
                    abortBtn.textContent = 'Abort Training';
                    if (lossChart) lossChart.update();
                    break;
            }
        };


        websocket.onclose = () => {
            addLog('Connection closed.');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Start Training';
            abortBtn.disabled = true;
            abortBtn.textContent = 'Abort Training';
            if (lossChart) lossChart.update();
        };

        websocket.onerror = (error) => {
            addLog('WebSocket Error.');
            console.error("WebSocket Error:", error);
            finalResultDiv.innerHTML = `<h3 style="color: red;">Connection Error</h3><p>Could not connect to WebSocket.</p>`;
            submitBtn.disabled = false;
            submitBtn.textContent = 'Start Training';
            abortBtn.disabled = true;
            abortBtn.textContent = 'Abort Training';
        };
    }

    // --- Event Listeners ---
    numLayersInput.addEventListener('input', updateLayerInputs);
    form.addEventListener('submit', handleTrain);

    // --- Initialization ---
    updateLayerInputs();
    prefillFromQuery();
    initChart();
});
