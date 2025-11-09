// main.js

// Espera a que todo el HTML esté cargado antes de ejecutar el script
document.addEventListener("DOMContentLoaded", () => {
    
    // --- Referencias al DOM ---
    const numLayersInput = document.getElementById('num_layers');
    const layersContainer = document.getElementById('layers-container');
    const lrSlider = document.getElementById('lr');
    const lrValueSpan = document.getElementById('lr-value');
    const form = document.getElementById('train-form');
    const submitBtn = document.getElementById('submit-btn');
    const logArea = document.getElementById('log-area');
    const finalResultDiv = document.getElementById('final-result');
    const chartCtx = document.getElementById('loss-chart').getContext('2d');
    
    // Nuevos inputs
    const numEpochsInput = document.getElementById('num_epochs');
    const batchSizeInput = document.getElementById('batch_size');

    // --- Variables Globales ---
    let lossChart;
    let websocket;
    let lossHistory = []; // Para la media móvil

    // --- Inicializar el gráfico ---
    function initChart() {
        if (lossChart) {
            lossChart.destroy();
        }
        lossHistory = []; // Limpia el historial para la media móvil
        
        lossChart = new Chart(chartCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Loss (Bruto)',
                        data: [], // Recibirá {x, y}
                        borderColor: 'rgba(75, 192, 192, 0.5)', // Color suave
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 1,
                        pointRadius: 0
                    },
                    {
                        label: 'Media Móvil (Tendencia)',
                        data: [], // Recibirá {x, y}
                        borderColor: 'rgba(235, 99, 132, 1)', // Color destacado
                        backgroundColor: 'rgba(235, 99, 132, 0.2)',
                        borderWidth: 2, // Más gruesa
                        pointRadius: 0
                    }
                ]
            },
            options: {
                scales: {
                    x: { 
                        type: 'linear', // Eje X numérico
                        title: { display: true, text: 'Pasos (Steps)' } 
                    },
                    y: { 
                        title: { display: true, text: 'Loss' }, 
                        type: 'logarithmic' // Eje Y logarítmico
                    }
                },
                animation: false
            }
        });
    }

    // --- Función para añadir a la bitácora ---
    function addLog(message) {
        logArea.textContent += message + '\n';
        logArea.scrollTop = logArea.scrollHeight; // Auto-scroll
    }

    // --- Función para actualizar el gráfico ---
    function updateChart(step, loss) {
        // 1. Añadir el loss bruto al historial
        lossHistory.push(loss);
        
        // 2. Calcular la ventana de la media móvil
        let windowSize = (step < 500) ? 10 : 50;
        windowSize = Math.min(windowSize, lossHistory.length);

        // 3. Calcular la media móvil
        let movingAverage = 0;
        if (lossHistory.length > 0) {
            const recentHistory = lossHistory.slice(-windowSize);
            const sum = recentHistory.reduce((a, b) => a + b, 0);
            movingAverage = sum / recentHistory.length;
        }

        // 4. Añadir ambos puntos a sus datasets
        // Dataset 0: Loss Bruto
        lossChart.data.datasets[0].data.push({ x: step, y: loss });
        
        // Dataset 1: Media Móvil
        if (movingAverage > 0) {
            lossChart.data.datasets[1].data.push({ x: step, y: movingAverage });
        }

        // 5. Actualizar el gráfico
        lossChart.update('none');
    }

    // --- Función para manejar el envío (WebSocket) ---
    function handleTrain(event) {
        event.preventDefault();
        submitBtn.disabled = true;
        submitBtn.textContent = 'Entrenando...';

        // Limpiar resultados anteriores
        initChart(); // Esta función ahora también limpia 'lossHistory'
        logArea.textContent = '';
        finalResultDiv.innerHTML = '';
        
        // 1. Recopilar toda la configuración
        const layerInputs = document.querySelectorAll('.layer-size');
        const hidden_layers_config = Array.from(layerInputs).map(input => parseInt(input.value));
        
        const config = {
            model_type: document.getElementById('model_type').value,
            optimizer: document.getElementById('optimizer').value,
            learning_rate: parseFloat(lrSlider.value),
            num_epochs: parseInt(numEpochsInput.value),
            batch_size: parseInt(batchSizeInput.value),
            num_layers: parseInt(numLayersInput.value),
            hidden_layers_config: hidden_layers_config
        };

        // 2. Conectar WebSocket
        const ws_url = `ws://${window.location.host}/ws/train`;
        websocket = new WebSocket(ws_url);

        websocket.onopen = () => {
            addLog('Conectado al servidor...');
            addLog('Enviando configuración...');
            websocket.send(JSON.stringify(config));
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch(data.type) {
                case 'log':
                    addLog(data.message);
                    break;
                case 'loss':
                    updateChart(data.step, data.loss);
                    break;
                case 'result':
                    addLog('¡Entrenamiento Completo!');
                    lossChart.update(); // Dibujo final del gráfico
                    
                    finalResultDiv.innerHTML = `
                        <h3>¡Resultados Finales!</h3>
                        <p><strong>Modelo:</strong> ${data.model_type}</p>
                        <p><strong>Optimizador:</strong> ${data.optimizer}</p>
                        <p><strong>Accuracy Final:</strong> ${data.accuracy.toFixed(2)}%</p>
                        <p><strong>Tiempo Total:</strong> ${data.training_time.toFixed(2)}s</p>
                        <img src="${data.plot_path}?t=${new Date().getTime()}" alt="Gráfico de pérdida final">
                    `;
                    break;
                case 'error':
                    addLog(`ERROR: ${data.message}`);
                    finalResultDiv.innerHTML = `<h3 style="color: red;">Error</h3><p>${data.message}</p>`;
                    break;
            }
        };

        websocket.onclose = () => {
            addLog('Conexión cerrada.');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Iniciar Entrenamiento';
            if (lossChart) lossChart.update(); // Asegura el último frame
        };

        websocket.onerror = (error) => {
            addLog('Error de WebSocket.');
            console.error("WebSocket Error:", error);
            finalResultDiv.innerHTML = `<h3 style="color: red;">Error de Conexión</h3><p>No se pudo conectar al WebSocket.</p>`;
        };
    }

    // --- Event Listeners (Formulario) ---
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
    
    numLayersInput.addEventListener('input', updateLayerInputs);
    lrSlider.addEventListener('input', () => {
        lrValueSpan.textContent = parseFloat(lrSlider.value).toExponential(2);
    });
    form.addEventListener('submit', handleTrain);

    // --- Inicialización ---
    updateLayerInputs();
    lrValueSpan.textContent = parseFloat(lrSlider.value).toExponential(2);
    initChart();
});