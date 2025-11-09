import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, hidden_structure=None):
        """
        Inicializa la red neuronal simple.
        
        Args:
            hidden_structure (list, optional): 
                Una lista de enteros que define el tamaño de cada capa oculta.
                Si es None, usará el valor por defecto [128, 64].
        """
        super(SimpleNN, self).__init__()
        
        # 1. Establece la estructura por defecto si no se proporciona una
        if hidden_structure is None:
            hidden_structure = [128, 64]
            
        self.flatten = nn.Flatten()
        
        # 2. Construye el bloque de capas ocultas dinámicamente
        layers = []
        in_features = 28 * 28  # El input siempre es 28*28
        
        for out_features in hidden_structure:
            # Agrega la capa lineal
            layers.append(nn.Linear(in_features, out_features))
            # Agrega la activación
            layers.append(nn.ReLU())
            # La entrada de la siguiente capa es la salida de esta
            in_features = out_features
            
        # 3. Usa nn.Sequential para empaquetar todas las capas ocultas
        # El * "desempaca" la lista de capas
        self.hidden_layers = nn.Sequential(*layers)
        
        # 4. La capa de salida se conecta a la última capa oculta
        # 'in_features' ahora tiene el tamaño de la última capa (ej: 64 o 32)
        self.output_layer = nn.Linear(in_features, 10) # El output siempre es 10

    def forward(self, x):
        x = self.flatten(x)
        # Pasa la data por todo el bloque de capas ocultas
        x = self.hidden_layers(x)
        # Pasa por la capa final de salida
        x = self.output_layer(x)
        return x
    

import torch.nn.functional as F

class ConvNN(nn.Module):
    def __init__(self, conv_channels=None, linear_hidden=None, 
                 input_dims=(1, 28, 28), num_classes=10):
        """
        Inicializa la red convolucional flexible.

        Args:
            conv_channels (list, optional): 
                Lista de canales de salida para cada bloque Conv+Pool.
                Ej: [16, 32] (por defecto).
            linear_hidden (list, optional): 
                Lista de tamaños para las capas ocultas lineales.
                Ej: [128] (por defecto).
            input_dims (tuple, optional): 
                Dimensiones de la imagen de entrada (Canales, Alto, Ancho).
                Ej: (1, 28, 28) para MNIST.
            num_classes (int, optional): 
                Número de clases de salida. Ej: 10.
        """
        super(ConvNN, self).__init__()

        # --- 1. Establecer valores por defecto ---
        if conv_channels is None:
            conv_channels = [16, 32]
        if linear_hidden is None:
            linear_hidden = [128]
            
        in_channels, h, w = input_dims
        self.num_classes = num_classes

        # --- 2. Construir bloque convolucional dinámico ---
        conv_layers = []
        for out_channels in conv_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels # La entrada de la sig. capa es la salida de esta
        
        # Usamos nn.Sequential para agrupar el bloque convolucional
        self.conv_block = nn.Sequential(*conv_layers)

        # --- 3. El truco: Calcular el tamaño de entrada lineal ---
        # Creamos un tensor falso con las dimensiones de entrada
        with torch.no_grad(): # No necesitamos calcular gradientes aquí
            dummy_input = torch.randn(1, *input_dims) # Ej: (1, 1, 28, 28)
            dummy_output = self.conv_block(dummy_input)
            # aplanamos el dummy_output para saber su tamaño
            # .numel() calcula el número total de elementos
            linear_in_features = dummy_output.numel() 
            
        # --- 4. Construir bloque lineal dinámico ---
        linear_layers = []
        for out_features in linear_hidden:
            linear_layers.append(nn.Linear(linear_in_features, out_features))
            linear_layers.append(nn.ReLU())
            linear_in_features = out_features # Actualizamos para la sig. capa
            
        self.classifier_block = nn.Sequential(*linear_layers)
        
        # --- 5. Capa de salida final ---
        # linear_in_features ahora tiene el tamaño de la última capa oculta (ej: 128)
        self.output_layer = nn.Linear(linear_in_features, self.num_classes)

    def forward(self, x):
        # x tiene forma (Batch, Canales, Alto, Ancho)
        x = self.conv_block(x)
        
        # Aplana la salida para las capas lineales
        x = x.view(x.size(0), -1) 
        
        # Pasa por el clasificador
        x = self.classifier_block(x)
        
        # Pasa por la capa de salida
        x = self.output_layer(x)
        return x