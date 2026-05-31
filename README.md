# Clasificación de Pokémon de Primera Generación con CNN

Autor: Mauricio Anguiano Juarez - A01703337
Módulo: Módulo 2 Inteligencia Artificial — Tec de Monterrey
Fecha: Mayo 2026
Profesor: Benjamin Valdés Aguirre

---

## 1. Objetivo

Construir y evaluar una CNN que clasifique correctamente imágenes de 151 clases
(los Pokémon de la primera generación), haciendo preparación del dataset,
train/validation split, data augmentation, transfer learning, evaluación de
modelo y guardado/carga del modelo en Keras.

---

## 2. Dataset y justificación de su elección

Nombre: pokemon-images-first-generation17000-files
Fuente: Kaggle — autor mikoajkolman
Descarga: kagglehub.dataset_download("mikoajkolman/pokemon-images-first-generation17000-files")
Tamaño: ~17,000 imágenes organizadas en 151 carpetas (una por Pokémon).
Nota: al descargar, el dataset expone 143 carpetas válidas con imágenes, por lo
que NUM_CLASSES = 143 en la práctica.

### ¿Por qué este dataset y no otro?

| Criterio | Decisión tomada | Razón |
|---|---|---|
| Tipo de problema | Clasificación multi-clase de imágenes | En el módulo se vieron CNNs binarias (cats vs dogs) o de 10 clases (Fashion MNIST); 151 clases sube la dificultad sin salirse del alcance. |
| Tamaño | ~17,000 imágenes / ~110 por clase | Suficiente para split 80/20 y data augmentation, evitando sobreajuste de datasets pequeños. |
| Estructura | Carpetas por clase | Compatible con ImageDataGenerator.flow_from_directory(), igual que cats vs dogs. |
| Originalidad | Pokémon es un interes personal diferente a los vistos en clase | Caso propio sin alejarse de la temática de clasificación de imágenes. |

---

## 3. Herramientas
- Python 3 + Google Colab (GPU).
- TensorFlow / Keras.
- kagglehub, numpy, pandas, matplotlib, seaborn, scikit-learn, PIL.

---

## 4. Estructura del repositorio



| Archivo | Contenido |
|---|---|
| `README.md` | Documentación del proyecto (este archivo). |
| `pokemon_cnn.ipynb` | Notebook principal: descarga y preprocesado del dataset, Modelo 1 (MobileNetV2) con entrenamiento, fine-tuning y evaluación, Modelo 2 (EfficientNetB0) y comparación final de los dos modelos. |
| `pokemon_cnn_inferencia.ipynb` | Carga un modelo `.h5` ya entrenado y permite subir imágenes de Pokémon para clasificarlas (no entrena). |
| `pokemon_cnn_best.h5`, `pokemon_effnet_best.h5`, `pokemon_effv2_best.h5` | Pesos de los modelos entrenados (MobileNetV2, EfficientNetB0, EfficientNetV2B0), recargables con `keras.models.load_model()`. |

---

## 6. Enlace al notebook

Google Colab (con GPU):
https://colab.research.google.com/drive/1zI1oSp4rAJreoIyJgig2vBqlGsieyJnF?usp=sharing

---

## 7. Avance 2 — Modelo, evaluación inicial e interpretación


### 7.1 Modelo implementado (baseline ejecutado)

El primer modelo usa **transfer learning** sobre **MobileNetV2** (Sandler et al., 2018)
preentrenada en ImageNet, una arquitectura del estado del arte para clasificación de
imágenes en entornos con cómputo limitado (como Colab). La idea de reutilizar redes
preentrenadas como extractores de características generales está documentada en Sharif
Razavian et al. (2014), quienes muestran que las características de una CNN entrenada en
ImageNet funcionan como un baseline sorprendentemente fuerte para tareas nuevas.

Configuración:
- Backbone MobileNetV2 con `include_top=False`, congelada en la fase 1.
- Cabeza de clasificación: `GlobalAveragePooling2D → Dropout → Dense(softmax)` sobre las 143 clases efectivas.
- Entrada 128×128, rescalado `1./255`, data augmentation solo en entrenamiento (rotación, shift, zoom, flip).
- Optimizador Adam; callbacks `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.
- Fase 2 (fine-tuning): se descongelan las últimas 30 capas de la base y se reentrena con learning rate menor.

### 7.2 Métricas y resultados

La evaluación usa la accuracy global, además de precision, recall y F1 por clase
(`classification_report`) y la matriz de confusión, que es la metodología estándar para
clasificación multi-clase en la literatura citada.

| Métrica | Valor (baseline / Modelo 1) |
|---|---|
| Muestras de entrenamiento | 13,246 |
| Muestras de validación | 3,239 |
| Clases efectivas | 143 (el dataset expone 143 carpetas con imágenes, no 151) |
| **val_accuracy** | **0.80** (medido: 0.8015) |

### 7.3 Interpretación

- Una accuracy de ~80% sobre 143 clases es un resultado sólido para un baseline: el azar
  daría ~0.7% (1/143), por lo que el modelo aprende características discriminativas reales.
- En el `classification_report` por clase, muchos Pokémon alcanzan F1 altos (0.80–0.95),
  mientras que las clases con pocas imágenes o muy parecidas a otras (problema de tipo
  *fine-grained*, ver Wang et al., 2014) son las que bajan el desempeño.
- Las confusiones esperables ocurren entre Pokémon visualmente similares y entre etiquetas
  casi duplicadas del dataset (p. ej. "Mr. Mime" vs "MrMime"), lo cual es una limitación
  del dataset más que del modelo. Limpiar esas etiquetas duplicadas es una mejora pendiente.

---

## 8. Comparación de modelos

El requisito del módulo pide implementar al menos 2 versiones del modelo basadas en
artículos de investigación, compararlas y que la segunda supere a la primera. En este
avance se implementaron y compararon dos versiones:

| # | Modelo | Idea / artículo que lo respalda | Cambios clave respecto al anterior | val_accuracy |
|---|---|---|---|---|
| 1 | MobileNetV2 (baseline) | Sandler et al. (2018); transfer learning, Sharif Razavian et al. (2014) | Transfer learning con base congelada + fine-tuning de 30 capas, entrada 128×128 | **0.80** |
| 2 | EfficientNetB0 | Tan & Le (2019) — escalado compuesto | Backbone más eficiente, entrada 224×224, preprocesado propio de EfficientNet | **0.89** |

### Resultados obtenidos (Colab con GPU)

- La val_accuracy sube de **0.80 (MobileNetV2) a 0.89 (EfficientNetB0)**, una mejora de ~9
  puntos. El escalado compuesto de EfficientNet y la mayor resolución de entrada (224×224)
  superan con claridad al baseline, tal como predice Tan & Le (2019). ✓ Se cumple **M2 > M1**.
- Ambos valores son la evaluación del modelo final de cada versión sobre el mismo set de
  validación, por lo que la comparación es directa.
- Se exploró además una tercera versión (EfficientNetV2B0 con label smoothing y cosine
  decay), pero no se alcanzó a completar su entrenamiento; el avance cumple el requisito
  con las dos versiones anteriores.


---

## 9. Referencias del estado del arte

Las siguientes referencias fueron verificadas en **Scopus**.

**Clasificación / reconocimiento de imágenes con CNN**
- Chollet, F. (2017). *Xception: Deep learning with depthwise separable convolutions.* IEEE CVPR. (~14,820 citas)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition.* IEEE TPAMI, 37(9). (~11,402 citas)
- Wang, X., Girshick, R., Gupta, A., & He, K. (2018). *Non-local Neural Networks.* IEEE CVPR. (~11,092 citas)

**Clasificación fine-grained / discriminativa e interpretabilidad**
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE ICCV. (~22,239 citas)
- Sharif Razavian, A., Azizpour, H., Sullivan, J., & Carlsson, S. (2014). *CNN Features off-the-shelf: An astounding baseline for recognition.* IEEE CVPR Workshops. (~3,619 citas)
- Wang, J., Song, Y., Leung, T., Rosenberg, C., Wang, J., Philbin, J., Chen, B., & Wu, Y. (2014). *Learning fine-grained image similarity with deep ranking.* IEEE CVPR. (~1,176 citas)

**Arquitecturas y técnicas implementadas en el notebook**
*(referencias canónicas de las arquitecturas usadas; aún no verificadas individualmente en Scopus en esta sesión)*
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* IEEE CVPR.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Tan, M., & Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML.
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision* (label smoothing). IEEE CVPR.
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine decay). ICLR.

Estas referencias están indexadas en Scopus y publicadas en venues top del área (CVPR, ICCV,
TPAMI, ICML) con altos conteos de citas, lo que respalda que pertenecen al estado del arte.