# Especificación Técnica: Heurística de Inicialización Dirigida (Warm-Starting) para CPPN

Este documento sirve como la **especificación matemática y de diseño algorítmico** para que el equipo de desarrollo implemente la Fase de Inicialización de la CPPN (Fase 4b). El objetivo es sustituir la inicialización aleatoria de la población (que condena al modelo a un CKA insostenible de ~0.13 debido a la "muerte cognitiva" de los pesos ruidosos) por un anclaje funcional guiado por los pesos del modelo preentrenado original.

---

## 1. El Objetivo del Warm-Starting

El objetivo principal es inicializar los parámetros $\theta$ de la CPPN de modo que el bloque generado emule de entrada el comportamiento del bloque denso original del profesor:

1.  **Fidelidad Inicial Excepcional:** Garantizar un alineamiento semántico inicial de **$\text{CKA} \ge 0.85$** y una pérdida proxy de validación **$\text{proxy\_nll} \le 4.0$** en la generación 0.
2.  **Preservación de la Inteligencia Base:** Evitar que el optimizador CMA-ES tenga que redescubrir las leyes del lenguaje o las representaciones de los tokens mediante muestreo ciego (búsqueda de orden cero), permitiéndole actuar estrictamente como un **"escultor" de la esparsidad estructural** ($D_{\text{arch}} \ge 0.4$).

---

## 2. Formulación Matemática de la Heurística

La CPPN se define como una función continua $f_{\text{CPPN}}(\mathbf{v}_{\text{in}}; \theta) \to [w_{ij}, l_{ij}]$, parametrizada por un conjunto de pesos de red neuronal $\theta$ ($\approx 32\text{K}$ parámetros). Queremos aproximar la matriz de pesos del bloque FFN denso original, $W_{\text{dense}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$.

Proponemos dos métodos de inicialización que el equipo de desarrollo puede implementar directamente en la infraestructura híbrida (Python/Rust):

### Método A: Optimización por Gradiente de la CPPN (Frobenius Regression)
Dado que la CPPN es una red extremadamente pequeña, podemos resolver una regresión de mínimos cuadrados no lineal sobre la norma de Frobenius utilizando autograd directamente en la fase de inicialización de Python (Fase 1/4) antes de transferir el control al loop evolutivo de Rust.

#### Función de Pérdida de Inicialización:
$$\min_{\theta} \mathcal{L}_{\text{init}}(\theta) = \frac{1}{2d_{\text{in}}d_{\text{out}}} \sum_{i=1}^{d_{\text{in}}} \sum_{j=1}^{d_{\text{out}}} \left( w_{ij}(\theta) - W_{\text{dense}, ij} \right)^2 + \frac{\lambda}{2} \|\theta\|_2^2$$

Donde:
*   $w_{ij}(\theta)$ es la primera salida de la CPPN para el vector de entrada de coordenadas $\mathbf{v}_{\text{in}}(i, j)$.
*   $W_{\text{dense}, ij}$ es el peso real del modelo preentrenado.
*   $\lambda$ es el coeficiente de regularización de Ridge (típicamente $10^{-4}$) para prevenir la saturación de los pesos de la CPPN.

#### Algoritmo de Inicialización (Fase de Arranque):
1.  **Muestreo del Sustrato:** Se evalúa de manera determinista el vector de coordenadas $\mathbf{v}_{\text{in}}$ para cada uno de los $d_{\text{in}} \times d_{\text{out}}$ pares de neuronas.
2.  **Ajuste por Épocas:** Se entrena la CPPN utilizando el optimizador Adam por un número reducido de iteraciones (ej. $200\text{--}500$ épocas). Dado que el tamaño de la CPPN es insignificante, esta regresión toma **menos de 3 segundos** en la CPU/GPU de Karuo.
3.  **Alineamiento del Enlace:** Se ajustan los sesgos (*biases*) de la segunda salida de la CPPN ($l_{ij}$) para garantizar que la probabilidad media de enlace inicial $\mathbb{E}[l_{ij}]$ sea equivalente al ratio de esparsidad inicial deseado, evitando que el umbral $\tau$ filtre conexiones de forma descontrolada en la generación 0.

---

### Método B: Proyección Analítica por Pseudoinversa (Moore-Penrose Dual)
Si el equipo busca un arranque instantáneo sin iteraciones de gradiente, se puede utilizar una aproximación basada en *Extreme Learning Machines* (ELM):

1.  **Congelación de Capas Ocultas:** Se inicializan las capas ocultas de la CPPN de forma aleatoria fija con funciones de activación periódicas y gaussianas, actuando como un proyector de características no lineales.
2.  **Extracción del Mapa de Activaciones ($H$):** Se computa el forward pass de la CPPN para todas las coordenadas de entrada, guardando la matriz de activaciones de la última capa oculta, $H \in \mathbb{R}^{(d_{\text{in}} \cdot d_{\text{out}}) \times d_{\text{hidden}}}$.
3.  **Resolución de la Capa de Salida ($\theta_{\text{out}}$):** Los pesos de la última capa de la CPPN ($\theta_{\text{out}}$) se calculan analíticamente en tiempo constante mediante la regularización de Tikhonov:
    $$\theta_{\text{out}} = \left( H^T H + \lambda I \right)^{-1} H^T \text{vec}(W_{\text{dense}})$$

---

## 3. Implementación de Referencia (Prototipo Python/PyTorch)

A continuación se muestra el código limpio y robusto que el equipo debe incorporar en `python/scripts/warm_start.py` para poblar el genoma base de la evolución:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class WarmCPPN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super().__init__()
        # Arquitectura alineada con HIDDEN = 16+16 de la v4 para evitar presión de registros
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sin(), # Activación periódica para inducir simetría de bloques
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, coords):
        return self.net(coords)

def compute_warm_start_weights(target_weights, d_in, d_out, epochs=300, lr=0.01):
    """
    Resuelve la regresión no lineal de Frobenius para anclar la CPPN a la matriz real del LLM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cppn = WarmCPPN().to(device)
    optimizer = optim.Adam(cppn.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 1. Generar sustrato de coordenadas determinista [-1, 1]
    x_i = torch.linspace(-1.0, 1.0, d_in, device=device)
    y_j = torch.linspace(-1.0, 1.0, d_out, device=device)
    
    grid_x, grid_y = torch.meshgrid(x_i, y_j, indexing='ij')
    delta_x = grid_y - grid_x # Flujo relativo
    delta_y = grid_y - grid_x
    
    # 2. Construir el vector de entrada de 8 dimensiones (Anexo A de la v4)
    v_in = torch.stack([
        grid_x, 
        grid_y, 
        grid_x, # x_j aproximado
        grid_y, # y_j aproximado
        delta_x,
        delta_y,
        torch.sin(torch.pi * grid_x),
        torch.cos(torch.pi * grid_y)
    ], dim=-1).view(-1, 8) # [d_in * d_out, 8]
    
    target_flat = target_weights.to(device).view(-1, 1) # [d_in * d_out, 1]
    
    # 3. Loop de Ajuste Rápido (Frobenius)
    cppn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = cppn(v_in) # [d_in * d_out, 2]
        w_pred = out[:, 0:1] # Primera salida: Pesos
        
        loss = criterion(w_pred, target_flat)
        loss.backward()
        optimizer.step()
        
    # Extraer el genoma optimizado como un tensor plano para el GGUF disperso
    flat_genome = torch.cat([p.data.view(-1) for p in cppn.parameters()])
    return flat_genome.cpu().numpy()
```

---

## 4. Estructuración de la Suite de Pruebas de Calidad (Fase 4b QA)

Para certificar que el anclaje se ha completado correctamente, el equipo de desarrollo debe añadir el siguiente test de aserción en `saor-domain::tests::warm_start`:

```rust
#[test]
fn test_warm_start_fidelity_threshold() {
    let original_layer_weights = load_original_ffn_tensor();
    let cppn_warm_genome = run_warm_start_regression(&original_layer_weights);
    
    // Decodificar la topología utilizando el kernel de OpenCL con esparsidad base tau = 0.0 (totalmente densa)
    let candidate_weights = decode_cppn_topology(&cppn_warm_genome, 0.0);
    
    // Evaluar fidelidad semántica usando CKA local (Paso 5)
    let cka_score = compute_local_cka(&original_layer_weights, &candidate_weights);
    
    // Asertar que el modelo inicial hereda la inteligencia del profesor
    assert!(
        cka_score >= 0.85, 
        "Fallo de Warm-Start: CKA inicial ({:.4}) por debajo del umbral de seguridad de 0.85", 
        cka_score
    );
}
```

---

## 5. Dinámica de Exploración durante la Evolución

Una vez que la población se inicializa alrededor de esta semilla optimizada, la evolución **no se vuelve estática ni se sesga hacia el conformismo**:

1.  **Diversidad vía Perturbación Controlada:** Los candidatos iniciales de la población de CMA-ES se generan añadiendo ruido gaussiano modulado $\mathcal{N}(0, \sigma^2 I)$ al genoma de la CPPN optimizada.
2.  **Exploración de Estructura:** El optimizador evolutivo se enfoca exclusivamente en ajustar la segunda salida ($l_{ij}$) y el umbral dinámico de esparsidad $\tau$. Al mutar estas variables, la evolución experimenta de forma no dirigida eliminando diferentes grupos de conexiones redundantes, mientras los pesos remanentes conservan su afinidad funcional original.
3.  **Convergencia del Subespacio Activo:** La perturbación evolutiva se proyecta únicamente sobre el subespacio latente de la diagonal de la Fisher, garantizando que el cómputo evolutivo se invierta solo en alterar las zonas más sensibles y dinámicas del flujo de activations del Transformer.
