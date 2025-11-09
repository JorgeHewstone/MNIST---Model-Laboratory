// Espera a que todo el HTML esté cargado
document.addEventListener("DOMContentLoaded", () => {

    // --- Referencias al DOM (Panel de Dibujo) ---
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('result');
    const modelInfoDiv = document.getElementById('model-info');

    // --- Referencias al DOM (Panel de Parámetros) ---
    const modelTypeSelect = document.getElementById('model_type');
    const optimizerSelect = document.getElementById('optimizer');
    const batchSizeSelect = document.getElementById('batch_size');
    const lrRangeSelect = document.getElementById('lr_range');
    const numLayersInput = document.getElementById('num_layers');
    const layersContainer = document.getElementById('layers-container');

    // --- Lógica del Canvas ---
    let drawing = false;
    ctx.lineWidth = 25;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    function clearCanvas() {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultDiv.textContent = '...';
        modelInfoDiv.innerHTML = '<p>Dibuja un dígito para predecir.</p>';
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
        // Ajustar por el escalado si la pantalla tiene zoom
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
    
    // Inicia el canvas en blanco
    clearCanvas();

    // --- Lógica del Formulario de Parámetros ---
    
    function updateLayerInputs() {
        const count = parseInt(numLayersInput.value) || 0;
        layersContainer.innerHTML = ''; 
        const defaults = [128, 64, 32, 16]; 
        for (let i = 0; i < count; i++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'layer-size';
            input.placeholder = `Capa ${i + 1}`;
            input.value = defaults[i] || 64;
            input.required = true;
            layersContainer.appendChild(input);
        }
    }

    // --- Lógica de Predicción ---

    async function predict() {
        resultDiv.textContent = 'Buscando...';
        modelInfoDiv.innerHTML = '';
        
        // 1. Crear imagen para la API
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        tempCanvas.getContext('2d').drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);
        const imageData = tempCanvas.toDataURL('image/png');
        
        // 2. Recopilar *todos* los parámetros del modelo
        const layerInputs = document.querySelectorAll('.layer-size');
        const hidden_layers_config = Array.from(layerInputs).map(input => parseInt(input.value));

        const modelParams = {
            model_type: modelTypeSelect.value,
            optimizer: optimizerSelect.value,
            learning_rate_range: lrRangeSelect.value,
            batch_size: parseInt(batchSizeSelect.value),
            hidden_layers_config: hidden_layers_config
        };

        // 3. Enviar a la API
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
                // "Modelo no entrenado" es un resultado esperado
                resultDiv.textContent = 'N/A';
                modelInfoDiv.innerHTML = `<p style="color: #dc3545;">${data.error}</p>`;
            } else {
                // ¡Éxito!
                resultDiv.textContent = `Predicción: ${data.prediction} (${data.confidence})`;
                modelInfoDiv.innerHTML = `
                    <p>
                        <strong>Modelo encontrado:</strong> ${data.found_model.id}<br>
                        <strong>Accuracy:</strong> ${data.found_model.accuracy}<br>
                        <strong>LR (real):</strong> ${data.found_model.learning_rate}
                    </p>
                `;
            }
        } catch (err) {
            resultDiv.textContent = 'Error';
            modelInfoDiv.innerHTML = '<p style="color: red;">Error en la conexión con la API.</p>';
            console.error(err);
        }
    }

    // --- Event Listeners ---
    
    // Canvas
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', endDraw); // Parar si el mouse sale

    // Controles
    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', predict);

    // Formulario de parámetros
    numLayersInput.addEventListener('input', updateLayerInputs);

    // --- Inicialización ---
    updateLayerInputs(); // Generar los inputs de capa al cargar
});