# 🤖 Fine-tuning de T5 para Generación de Resúmenes
Este repositorio contiene la implementación del pipeline completo para el *fine-tuning* del modelo **T5 (t5-small)** sobre el dataset **CNN/DailyMail** (versión 3.0.0) para la tarea de generación automática de resúmenes (abstractive summarization).
Este proyecto fue desarrollado como el Entregable 2 para el curso de **Deep Learning (1INF52)**.
## 👥 Equipo de Desarrollo
* Aarón Ulises Santillán Huamán
* Joseph Alvaro Tintaya Marce
* Leoncio Percy Huaman Lopez
* Rodrigo Jose Valle Saavedra
## 🚀 Características Técnicas
* **Modelo Base:** `t5-small` cargado mediante la librería Hugging Face `transformers`.
* **Dataset:** Subconjunto optimizado del corpus `cnn_dailymail` (10,000 ejemplos de entrenamiento y 1,000 de validación).
* **Preprocesamiento:** Tokenización con truncamiento (entrada máxima de 512 tokens y salida de 128 tokens) agregando el prefijo de tarea `"summarize: "`.
* **Estrategia de Entrenamiento:** Configurado para 3 épocas utilizando precisión mixta acelerada por GPU (`fp16=True`), el optimizador `adafactor` y una tasa de aprendizaje de `2e-4`.
## 📊 Resultados de Evaluación (Métricas ROUGE)
El desempeño del modelo ajustado fue evaluado mediante la métrica ROUGE generando resúmenes con *Beam Search* (num_beams=4). Se obtuvieron los siguientes resultados en el conjunto de validación:
* **ROUGE-1:** 39.71%
* **ROUGE-2:** 18.41%
* **ROUGE-L:** 28.25%
* **ROUGE-Lsum:** 28.24%
## 🛠️ Stack Tecnológico
* `torch` (PyTorch con soporte CUDA)
* `transformers` y `datasets` (Hugging Face)
* `evaluate` y `rouge_score` (Cálculo de superposición léxica)
* `accelerate`