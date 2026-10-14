# T5 Summarization — Resúmenes automáticos de noticias

Fine-tuning del modelo **T5-small** sobre el dataset **CNN/DailyMail** para generar resúmenes abstractivos de noticias en inglés: el modelo no copia oraciones del texto, sino que redacta un resumen propio. Incluye el pipeline completo de carga de datos, tokenización, entrenamiento en GPU, evaluación con métricas ROUGE y una demo interactiva publicada en Hugging Face Spaces. Proyecto académico del curso de Deep Learning, desarrollado en un equipo de 4 personas.

**Demo en vivo:** [huggingface.co/spaces/Joseph1345/demo-ta-deeplearning](https://huggingface.co/spaces/Joseph1345/demo-ta-deeplearning)

> Si la demo está dormida por inactividad, presiona **"Restart this Space"** y espera alrededor de un minuto.

---

## Tecnologías Utilizadas

**Modelo:** T5-small (Hugging Face Transformers)

**Entrenamiento:** PyTorch (CUDA), Hugging Face `Trainer`, `datasets`, `accelerate`

**Evaluación:** `evaluate`, `rouge_score`

**Demo:** Gradio, Hugging Face Spaces

**Entorno:** Google Colab (GPU NVIDIA T4), Weights & Biases para el registro del entrenamiento

---

## Características

- **Resumen abstractivo** de artículos de noticias en inglés, con el prefijo de tarea `summarize:` propio de T5.
- **Pipeline completo y reproducible** en un solo notebook: datos, preprocesamiento, entrenamiento, evaluación y ejemplos cualitativos.
- **Evaluación cuantitativa** con ROUGE-1, ROUGE-2, ROUGE-L y ROUGE-Lsum sobre el conjunto de validación.
- **Demo web** donde cualquier persona pega un artículo y obtiene el resumen generado en tiempo real.

---

## El Proceso

**Datos.** CNN/DailyMail tiene más de 300,000 artículos con resúmenes escritos por periodistas. Para entrenar dentro de los límites de Google Colab usamos un subconjunto aleatorio de 10,000 artículos de entrenamiento. Cada entrada se tokenizó con un máximo de 512 tokens y cada resumen con 128, anteponiendo el prefijo `summarize:` que T5 usa para identificar la tarea.

**Entrenamiento con memoria limitada.** La GPU T4 de Colab tiene 16 GB, así que el entrenamiento se diseñó para no desbordarla: precisión mixta (`fp16`), *gradient checkpointing*, acumulación de gradientes (lotes efectivos de 4) y el optimizador Adafactor, que consume menos memoria que Adam. Entrenamos 3 épocas con una tasa de aprendizaje de 2e-4, en unos 55 minutos.

**Evaluación.** Generamos los resúmenes con *beam search* (4 haces) para todo el conjunto de validación y los comparamos con los resúmenes de referencia mediante ROUGE. Además revisamos ejemplos de forma cualitativa para detectar errores que las métricas no capturan, como detalles incorrectos en el resumen.

**Demo.** Publicamos el modelo entrenado en una aplicación de Gradio alojada en Hugging Face Spaces, para que se pueda probar desde el navegador sin instalar nada.

---

## Resultados

| Métrica | Resultado |
|---|---|
| ROUGE-1 | 39.71 % |
| ROUGE-2 | 18.41 % |
| ROUGE-L | 28.25 % |
| ROUGE-Lsum | 28.24 % |

Para un modelo pequeño (60 millones de parámetros) entrenado con una fracción del dataset, los resultados son competitivos con lo reportado para T5-small en CNN/DailyMail.

**Ejemplo real del conjunto de validación:**

> **Resumen humano:** Zully Broussard decided to give a kidney to a stranger. A new computer program helped her donation spur transplants for six kidney patients.
>
> **Resumen del modelo:** Zully Broussard gave one of her kidneys to a stranger. The power that multiplied her gift was data processing of genetic profiles from donor-recipient pairs. The chain of surgeries is to be wrapped up Friday.

El detalle de la metodología está en el [paper del proyecto](FineTuningDeT5ParaGeneraciondeTextos_Paper.pdf).

---

## Lo Que Aprendí

Aprendí a hacer fine-tuning de un modelo de lenguaje con recursos limitados, combinando técnicas de ahorro de memoria (fp16, gradient checkpointing, Adafactor), y a evaluar un modelo generativo más allá de una sola métrica, revisando también sus errores cualitativos.

---

## Cómo Puede Mejorarse

- **Modelos más grandes:** T5-base o T5-large logran resúmenes de mayor calidad si los recursos lo permiten.
- **Más datos:** entrenar con el dataset completo en lugar de un subconjunto de 10,000 artículos.
- **Métricas semánticas:** complementar ROUGE con BERTScore, que mide similitud de significado y no solo coincidencia de palabras.
- **Consistencia factual:** analizar y reducir los casos donde el resumen incluye detalles que no están en el artículo.

---

## Cómo Correr el Proyecto

1. Abre `FineTuningDeT5ParaGeneraciondeTextos.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Activa la GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*.
3. Ejecuta las celdas en orden. La primera instala las dependencias:

```bash
pip install transformers datasets sentencepiece evaluate rouge_score accelerate
```

El entrenamiento completo toma alrededor de una hora en una GPU T4.

---

## Equipo

Aarón Ulises Santillán Huamán · Joseph Alvaro Tintaya Marce · Leoncio Percy Huaman Lopez · Rodrigo Jose Valle Saavedra
