# Sesgos demográficos en modelos de visión supervisados y auto-supervisados

Análisis comparativo de **ResNet-50**, **DINOv2** y **SimCLR** (ResNet-50 contrastivo entrenado en ImageNet-1k) sobre el dataset **[FairFace](https://github.com/joojs/fairface)**, tomando como punto de partida el trabajo de Sirotkin et al. (2022).

La tarea evaluada es la **clasificación de género** a partir de embeddings congelados, y se mide cómo varía el comportamiento del modelo entre **grupos raciales**.

## Preguntas de investigación

1. ¿Qué rendimiento (Accuracy, F1) alcanza cada modelo sobre FairFace?
2. ¿Qué tan separadas quedan las razas en el espacio de embeddings (versión simplificada de iEAT)?
3. ¿Cómo se distribuyen *Demographic Parity* y *Equalized Odds* entre grupos raciales?
4. ¿Se comportan distinto los modelos supervisados y los auto-supervisados?

## Metodología

| Elemento | Detalle |
|---|---|
| Dataset | FairFace 0.25 (HuggingFace `HuggingFaceM4/FairFace`), split `validation`, primeras 2 500 imágenes |
| Modelos | ResNet-50 supervisado (torchvision), DINOv2-small (`facebook/dinov2-small`), SimCLR ResNet-50 (`niels-c/resnet50_simclr_linhead`, solo el backbone) |
| Representación | Embeddings congelados, normalizados L2 (sin fine-tuning) |
| Clasificador | Regresión logística sobre los embeddings; partición 70/30 estratificada por raza y género, métricas sobre el 30 % de test |
| Rendimiento | Accuracy, F1 ponderado |
| Equidad | Demographic Parity, TPR/FPR (Equalized Odds) y accuracy por grupo racial |
| Sesgo en embeddings | iEAT simplificado: distancia coseno media entre centroides raciales |
| Interpretabilidad | Grad-CAM (ResNet-50 y SimCLR, sobre la decisión del clasificador de género) |

## Estructura del repositorio

```
.
├── equity_bias_analysis.ipynb   # análisis completo, de principio a fin
├── requirements.txt
├── LICENSE
└── README.md
```

## Cómo ejecutarlo

```bash
git clone https://github.com/ibon44/fairface-ssl-bias-audit.git
cd fairface-ssl-bias-audit

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab notebooks/equity_bias_analysis.ipynb
```

- El notebook se entrega **sin salidas**: ejecútalo de arriba abajo (≈10–15 min en una T4) para generar resultados y redactar la sección de conclusiones.
- El notebook se desarrolló en **Google Colab con GPU T4**; en CPU funciona pero es considerablemente más lento.
- FairFace y los pesos de los modelos se descargan automáticamente en la primera ejecución (requiere conexión a internet).

## Estado del proyecto y limitaciones conocidas

Este es un proyecto exploratorio y los resultados deben leerse como **preliminares**:

- Subconjunto reducido (2 500 imágenes): no permite conclusiones poblacionales y no se reportan intervalos de confianza.
- Con ≈750 imágenes de test (≈100 por grupo racial), las diferencias pequeñas entre modelos pueden ser ruido; no se reportan aún intervalos de confianza.
- El iEAT es una **simplificación** (distancia entre centroides raciales); no es el iEAT original de Steed y Caliskan, que usa conjuntos de atributos y tamaños de efecto.
- Grad-CAM solo se aplica a arquitecturas convolucionales; DINOv2 (ViT) queda fuera.
- Se evalúa una sola tarea (género) y un solo atributo sensible (raza).

## Referencias

1. Sirotkin, K., Carballeira, P., & Escudero-Viñolo, M. (2022). *A study on the distribution of social biases in self-supervised learning visual models.* CVPR.
2. Wang, Z. et al. (2020). *Towards fairness in visual recognition: Effective strategies for bias mitigation.* CVPR.
3. Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). *Semantics derived automatically from language corpora contain human-like biases.* Science, 356(6334).
4. Kärkkäinen, K., & Joo, J. (2021). *FairFace: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation.* WACV.

## Licencia

[MIT](LICENSE) © 2026 Ibon de Mingo
