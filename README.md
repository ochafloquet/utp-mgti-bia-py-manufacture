# 🏭 Análisis de Manufactura Industrial — UTP MGTI BIA

**Universidad Tecnológica del Perú**  
Maestría en Gerencia de TI y Transformación Digital  
Business Intelligence & Analytics — Proyecto Final

---

## 📋 Descripción

Proyecto de analítica predictiva multi-técnica aplicado al sector manufactura industrial peruano. Analiza 1,000 registros operativos de 180 máquinas en plantas de Lima, Arequipa y Trujillo para:

- Predecir fallas de maquinaria con anticipación
- Segmentar máquinas según comportamiento y riesgo operativo
- Identificar factores críticos que impactan productividad
- Generar recomendaciones accionables para mantenimiento preventivo

## 🔬 Técnicas Aplicadas

| Técnica | Objetivo | Resultado |
|---------|----------|-----------|
| Regresión Logística | Predecir fallas (0/1) | Accuracy: 69.5%, AUC-ROC: 0.761 |
| K-Means Clustering | Segmentar máquinas | 4 segmentos: Eficientes/Estándar/Alerta/Críticas |
| Árbol de Decisión | Reglas interpretables | Top feature: vibración (45%) |
| Regresión Lineal | Factores de producción | Variables independientes de producción |
| Análisis de Correlación | Relaciones entre variables | Temp. y vibración más correlacionadas con fallas |

## 📁 Estructura del Repositorio

```
utp-mgti-bia-py-manufacture/
├── data/
│   └── manufactura/
│       ├── dataset_manufactura_1000.csv    ← Dataset principal
│       ├── generar_dataset.py              ← Generador (si no tienes el CSV)
│       ├── diccionario_datos_manufactura.xlsx
│       └── Indicaciones_Proyecto_Final.pdf
├── scripts/
│   ├── analisis_manufactura.py             ← Script principal de análisis
│   └── dashboard_manufactura.py            ← Dashboard Streamlit
├── resultados/
│   └── manufactura/                        ← Generado automáticamente
│       ├── dataset_limpio.csv
│       ├── maquinas_alto_riesgo.csv
│       ├── segmentos_clustering.csv
│       ├── metricas_modelos.csv
│       ├── resumen_ejecutivo.csv
│       ├── viz_01_*.png ... viz_08_*.png   ← Visualizaciones EDA
│       └── modelo_01_*.png ... modelo_04_*.png ← Gráficos de modelos
├── docs/
│   ├── convert_informe_to_html.py          ← Conversor MD→HTML→PDF
│   ├── INFORME_MANUFACTURA.md              ← Informe académico
│   └── INFORME_MANUFACTURA.html            ← Informe en HTML (generado)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Instrucciones de Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Preparar datos

Coloca el archivo `dataset_manufactura_1000.csv` en `data/manufactura/`.

Si no tienes el archivo, genera un dataset sintético de ejemplo:
```bash
python data/manufactura/generar_dataset.py
```

### 3. Ejecutar análisis principal

```bash
python scripts/analisis_manufactura.py
```

Esto genera automáticamente en `resultados/manufactura/`:
- 8 visualizaciones exploratorias (PNG)
- 4 gráficos de modelos (PNG)
- Modelos entrenados (PKL)
- CSVs de resultados

### 4. Lanzar dashboard interactivo

```bash
streamlit run scripts/dashboard_manufactura.py
```

Abre el navegador en `http://localhost:8501`

### 5. Generar informe en HTML/PDF

```bash
python docs/convert_informe_to_html.py
```

## 📊 Resultados Principales

- **Tasa de falla:** 38.5% del parque de maquinaria
- **Máquinas de alto riesgo:** 242 equipos con probabilidad >70%
- **Variables más críticas:** temperatura, vibración, días sin mantenimiento
- **Mejor predictor de fallas:** Regresión Logística (AUC-ROC: 0.761)
- **Segmento crítico:** 26% de máquinas en estado "Alerta" o "Críticas"

## 🛠️ Tecnologías

- **Python 3.10+**
- pandas, numpy — Manipulación de datos
- scikit-learn — Modelos de ML
- statsmodels — Estadística inferencial
- matplotlib, seaborn — Visualizaciones estáticas
- plotly, streamlit — Dashboard interactivo
- joblib — Persistencia de modelos

## 👥 Equipo

- [Nombre del Integrante 1]
- [Nombre del Integrante 2]
- [Nombre del Integrante 3]
- [Nombre del Integrante 4]

---

*UTP — Agosto 2025*
