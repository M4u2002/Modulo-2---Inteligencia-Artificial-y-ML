# Clasificación de Pokémon de Primera Generación con CNN

Autor: Mauricio Anguiano Juarez - A01703337
Módulo: Módulo 2 Inteligencia Artificial — Tec de Monterrey
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
| `PokemonCNN.ipynb` | Notebook principal: descarga y preprocesado del dataset, Modelo 1 (MobileNetV2) con entrenamiento, fine-tuning y evaluación, Modelo 2 (EfficientNetB0) y comparación final de los dos modelos. |
| `pokemon_cnn_pruebas.ipynb` | Demo de predicción: carga los modelos `.h5` ya entrenados y clasifica imágenes (no entrena); corre **los 2 modelos** lado a lado. |
| `pokemon_cnn_best.h5` | Pesos del Modelo 1 (MobileNetV2, 128×128, val_acc 0.81). |
| `pokemon_effnet_best.h5` | Pesos del Modelo 2 (EfficientNetB0, 224×224, mejor checkpoint 0.92). |
| `matriz_confusion.png` | Matriz de confusión del Modelo 1 generada por el notebook. |

---


## 5. Enlace al notebook

Google Colab (con GPU):
https://colab.research.google.com/drive/1zI1oSp4rAJreoIyJgig2vBqlGsieyJnF?usp=sharing

---

## 6. Avance 2 — Modelo, evaluación inicial e interpretación


### 6.1 Modelo implementado (baseline ejecutado)

El primer modelo usa **transfer learning** sobre **MobileNetV2** (Sandler et al., 2018)
preentrenada en ImageNet, una arquitectura del estado del arte para clasificación de
imágenes en entornos con cómputo limitado (como Colab). La idea de reutilizar redes
preentrenadas como extractores de características generales está documentada en Sharif
Razavian et al. (2014), quienes muestran que las características de una CNN entrenada en
ImageNet funcionan como un baseline sorprendentemente fuerte para tareas nuevas.

Configuración:
- Backbone MobileNetV2 con `include_top=False`, congelada en la fase 1.
- Cabeza de clasificación: `GlobalAveragePooling2D → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → Dense(143, softmax)` sobre las 143 clases efectivas.
- Entrada 128×128, rescalado `1./255`, data augmentation solo en entrenamiento (rotación, shift, zoom, flip).
- Optimizador Adam; callbacks `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.
- Fase 2 (fine-tuning): se descongelan las últimas 30 capas de la base y se reentrena con learning rate menor.

### 6.2 Métricas y resultados

La evaluación usa la accuracy global, además de precision, recall y F1 por clase
(`classification_report`) y la matriz de confusión, que es la metodología estándar para
clasificación multi-clase.

| Métrica | Valor (baseline / Modelo 1) |
|---|---|
| Muestras de entrenamiento | 13,246 |
| Muestras de validación | 3,239 |
| Clases efectivas | 143 (el dataset expone 143 carpetas con imágenes, no 151) |
| **val_accuracy** | **0.81** (medido: 0.8114) |
| macro-F1 / weighted-F1 | 0.80 / 0.81 (precision macro 0.82) |

### 6.3 Interpretación

- Un accuracy de ~81% sobre 143 clases es un resultado sólido para un baseline: el azar
  daría ~0.007% (1/143), por lo que el modelo aprende características discriminativas reales.
- En el `classification_report` por clase, muchos Pokémon alcanzan F1 altos (0.80–0.95),
  mientras que las clases con pocas imágenes o muy parecidas a otras (problema de tipo
  *fine-grained*) son las que bajan el desempeño.
- Las confusiones esperables ocurren entre Pokémon visualmente similares y entre etiquetas
  casi duplicadas del dataset (p. ej. "Mr. Mime" vs "MrMime"), lo cual es una limitación
  del dataset más que del modelo.

### 6.4 Matriz de confusión (Modelo 1)

La celda 13 del notebook calcula la matriz de confusión completa de 143×143, la guarda como
imagen (`matriz_confusion.png`, normalizada por fila) y lista las confusiones concretas. Como
una matriz de 143 columnas no es legible con números, lo más informativo es el listado de
**clases con más errores** y de **pares (real → predicho)** que más se confunden.

**Top de Pokémon con más errores (de la corrida real, Modelo 1):**

| Pokémon (real) | Errores / muestras |
|---|---|
| Pinsir | 13 / 26 |
| Electabuzz | 13 / 24 |
| Farfetch'd | 12 / 25 |
| Starmie | 12 / 28 |
| Rapidash | 12 / 25 |
| Pidgey | 12 / 38 |
| Pidgeotto | 12 / 24 |
| Marowak | 11 / 18 |
| Primeape | 11 / 27 |
| Kadabra | 11 / 17 |

**Confusiones concretas más frecuentes (real → predicho):** 
- Starmie → Staryu (9 veces)
- Rapidash → Ponyta (9)
- Pidgeotto → Pidgeot (7)
- Wartortle → Squirtle (5)
- Electrode → Voltorb (5)
- Poliwhirl → Poliwrath (5)
- Jolteon → Zapdos (4).

Casi todas las confusiones caen entre **el mismo Pokémon en distinta etapa
evolutiva** o **Pokémon del mismo tipo y silueta** . Es el problema clásico de clasificación *fine-grained*: distinguir clases visualmente casi idénticas. 

![Matriz de confusión del Modelo 1](matriz_confusion.png)

---

## 7. Comparación de modelos
Se nos pide implementar al menos 2 versiones del modelo basadas en
artículos de investigación, compararlas y que la segunda supere a la primera. En este
avance se implementaron y compararon dos versiones:

| # | Modelo | Idea / artículo que lo respalda | Cambios clave respecto al anterior | val_accuracy |
|---|---|---|---|---|
| 1 | MobileNetV2 (baseline) | Sandler et al. (2018) | Transfer learning con base congelada + fine-tuning de 30 capas, entrada 128×128 | **0.81** |
| 2 | EfficientNetB0 | Tan & Le (2019) — escalado compuesto | Backbone más eficiente, entrada 224×224, preprocesado propio de EfficientNet | **0.89** |

### Resultados obtenidos (Colab con GPU)

- El accuracy sube de **0.81 (MobileNetV2) a 0.89 (EfficientNetB0)**, una mejora de ~8 puntos. El escalado compuesto de EfficientNet y la mayor resolución de entrada (224×224) superan con claridad al baseline.
- Ambos valores son la evaluación del **modelo final** de cada versión sobre el mismo set de validación, por lo que la comparación es directa. (El **mejor checkpoint** de EfficientNetB0 el que guarda `pokemon_effnet_best.h5` alcanzó 0.92; se reporta el 0.89 del modelo final para comparar de forma equivalente con MobileNetV2, que también se reporta con su modelo final.) 

---

## 9. Referencias

> Solo se listan referencias que corresponden a algo **realmente implementado y usado** . Las arquitecturas se instancian directamente en el código (`tf.keras.applications.MobileNetV2` y `EfficientNetB0`), por lo que sus artículos originales (referencias *canónicas*) son los que se citan.

**Arquitecturas implementadas en el código:**
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2:
  Inverted Residuals and Linear Bottlenecks.* IEEE/CVF CVPR. — backbone del **Modelo 1**
  (`tf.keras.applications.MobileNetV2`).
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks.* ICML. — backbone del **Modelo 2** (`tf.keras.applications.EfficientNetB0`).