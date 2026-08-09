# INFORME EJECUTIVO: OPTIMIZACIÓN DE OPERACIONES INDUSTRIALES MEDIANTE ANALÍTICA PREDICTIVA MULTI-TÉCNICA

## PORTADA

**Universidad Tecnológica del Perú**  
**Maestría en Gerencia de TI y Transformación Digital**  
**Curso:** Business Intelligence & Analytics — Sesión 60271  
**Proyecto Final:** Optimización de operaciones industriales mediante analítica predictiva multi-técnica  
**Sector de aplicación:** Manufactura industrial en Lima, Arequipa y Trujillo, Perú  
**Integrantes:**  
- [Nombre del Integrante 1]  
- [Nombre del Integrante 2]  
- [Nombre del Integrante 3]  
- [Nombre del Integrante 4]  
**Docente:** [Nombre del Docente]  
**Fecha:** Agosto 2025  
**Lugar:** Lima, Perú

---

## RESUMEN EJECUTIVO

El presente informe desarrolla un proyecto académico de Business Intelligence & Analytics orientado al sector manufacturero industrial peruano, con cobertura en plantas ubicadas en Lima, Arequipa y Trujillo. El problema central analizado es una tasa de falla de maquinaria de **38.5%**, equivalente a 385 eventos de falla sobre un total de 1,000 registros operacionales capturados en **180 máquinas**. Esta magnitud de incidencia implica un nivel significativo de paradas no planificadas, afectación de la continuidad productiva, incremento de costos de mantenimiento correctivo y presión sobre la calidad operativa.

Con el propósito de transformar datos operativos en decisiones accionables, se aplicó la metodología **CRISP-DM**, estructurando el trabajo en seis fases: entendimiento del negocio, entendimiento de los datos, preparación de datos, modelado, evaluación y despliegue analítico. Sobre esta base metodológica, se implementaron cuatro técnicas complementarias en Python usando **pandas, scikit-learn, seaborn, matplotlib y streamlit**: **regresión logística binaria** para predecir fallas, **clustering K-Means** para segmentar perfiles operativos, **árbol de decisión** para generar reglas interpretables y **regresión lineal múltiple** para explorar la relación entre variables operativas y producción.

El dataset evaluado contiene **15 variables**, integrando identificadores, atributos categóricos del contexto operativo y mediciones numéricas de desempeño. Entre las variables más relevantes destacan la temperatura, vibración, presión, horas de operación, antigüedad, días desde el último mantenimiento, tasa de defectos, consumo energético y producción. En términos de calidad, los datos presentaron una condición favorable para el modelado: **0 valores nulos** y **0 registros duplicados**. La fase de preparación incluyó codificación de variables categóricas, estandarización para los modelos sensibles a escala y partición de entrenamiento/prueba de **80%/20%**, con estratificación en los modelos de clasificación.

Los resultados confirman que la analítica predictiva puede aportar valor concreto a la gestión del mantenimiento. El mejor desempeño correspondió a la **regresión logística**, con **Accuracy de 69.5%**, **Precision de 59.3%**, **Recall de 66.2%**, **F1 de 62.6%**, **AUC-ROC de 0.761** y validación cruzada de **73.0%**. En lenguaje de negocio, esto significa que el modelo logra discriminar razonablemente entre condiciones normales y condiciones de falla, identificando aproximadamente **66 de cada 100 fallas reales**. Complementariamente, el **árbol de decisión** obtuvo **Recall de 70.1%** y **AUC-ROC de 0.650**, con menor exactitud global (**60.0%**), pero mayor interpretabilidad, mostrando que la **vibración (44.9%)**, los **días desde mantenimiento (24.9%)** y la **temperatura (15.9%)** son los factores más determinantes.

El análisis de correlaciones refuerza estos hallazgos. Las relaciones más altas con la variable objetivo `falla_maquina` fueron **temperatura_c (0.303)**, **vibracion_mm_s (0.298)** y **dias_desde_mantenimiento (0.271)**. A nivel operativo, también se observaron diferencias por turno: la tasa de falla fue **42.86% en turno tarde**, **40.44% en turno noche** y **32.46% en turno mañana**, lo que sugiere un patrón de riesgo vinculado con carga operativa, fatiga, continuidad de uso o menor oportunidad de intervención preventiva durante ciertos horarios.

La segmentación con **K-Means (k = 4)**, aunque mostró un **Silhouette de 0.082** —indicando separación estadística débil entre clusters—, permitió construir una clasificación gerencial útil: **Eficientes (22.9% de falla)**, **Estándar (32.0%)**, **Alerta (47.0%)** y **Críticas (57.5%)**. Esta tipología facilita priorizar acciones diferenciadas según perfil operativo. Asimismo, el sistema identificó **242 registros de alto riesgo** con probabilidad estimada de falla superior al 70%, asociados a **137 máquinas únicas**, lo cual constituye una base concreta para campañas focalizadas de mantenimiento.

Finalmente, la **regresión lineal múltiple** sobre producción obtuvo **R² = -0.009** y **RMSE = 181 unidades**, evidenciando que la producción no mantiene una relación lineal significativa con las variables operativas incluidas. Lejos de ser un hallazgo negativo, este resultado revela que la productividad industrial depende también de factores no capturados en el dataset, tales como habilidad del operador, calidad de materia prima, cambios de lote, programación de producción, paradas logísticas, mix de productos y disciplina operativa.

Como conclusión ejecutiva, el proyecto demuestra que la organización puede pasar de una gestión reactiva a una gestión preventiva basada en datos. Las principales recomendaciones son: intervenir prioritariamente las **242 observaciones de alto riesgo**, redefinir frecuencias de mantenimiento cuando la vibración supere **5.45 mm/s** o el equipo acumule más de **165 días** sin mantenimiento, implantar un esquema de alertas por riesgo, y ampliar la captura de variables para mejorar modelos de producción y mantenimiento. En términos de negocio, incluso una reducción parcial de fallas en los segmentos **Alerta** y **Críticas** podría generar mejoras relevantes en disponibilidad, confiabilidad y costo total de operación.

## 1. INTRODUCCIÓN

### 1.1 Contexto Industrial

La manufactura industrial enfrenta actualmente una presión creciente por operar con alta disponibilidad, bajos niveles de desperdicio, costos controlados y calidad consistente. En ese contexto, la gestión de activos productivos se vuelve una capacidad estratégica, dado que una falla de máquina no solo interrumpe el flujo productivo, sino que también altera programas de entrega, genera reprocesos y reduce la eficiencia global del sistema.

En el caso analizado, la operación se distribuye en tres plantas peruanas: **Lima, Arequipa y Trujillo**. El parque industrial comprende **180 máquinas** pertenecientes a cinco categorías operativas: **Prensa, Torno CNC, Inyectora, Empacadora y Compresor**. Esta diversidad de activos hace necesario adoptar un enfoque analítico capaz de capturar patrones compartidos y, al mismo tiempo, diferencias entre tipos de equipo, ubicación y turno.

### 1.2 Problema de Negocio

El problema de negocio se centra en una tasa de falla de **38.5%**, valor elevado para una operación industrial que depende de la continuidad del servicio de maquinaria. Este nivel de incidencia sugiere una exposición material a:

- paradas no planificadas;
- incremento del mantenimiento correctivo;
- afectación de la productividad y del cumplimiento de órdenes;
- posible elevación de defectos y merma;
- consumo energético ineficiente por degradación operativa.

Por ello, la pregunta analítica principal es: **¿qué variables operativas explican mejor las fallas de maquinaria y cómo pueden utilizarse para priorizar mantenimiento y mejorar decisiones gerenciales?**

### 1.3 Objetivos del Análisis

**Objetivo general:**  
Desarrollar un análisis de Business Intelligence & Analytics basado en CRISP-DM para comprender, predecir y priorizar el riesgo de falla de maquinaria en una operación manufacturera industrial.

**Objetivos específicos:**

1. Caracterizar el comportamiento del dataset y evaluar su calidad.
2. Identificar variables con mayor relación con la falla de maquinaria.
3. Construir un modelo de clasificación para predecir fallas.
4. Segmentar perfiles operativos mediante clustering.
5. Obtener reglas interpretables de decisión para mantenimiento.
6. Evaluar si la producción se explica linealmente por variables operativas.
7. Traducir hallazgos analíticos a recomendaciones gerenciales accionables.

### 1.4 Alcance y Limitaciones

El análisis cubre **1,000 registros operativos**, tres plantas y cinco tipos de máquina. El alcance incluye modelado supervisado y no supervisado, interpretación de métricas y diseño conceptual de un sistema de alertas.

Las principales limitaciones son:

- el dataset representa una foto analítica y no una serie temporal continua;
- la variable de producción no incorpora factores exógenos relevantes;
- el clustering presenta separación estadística baja;
- los resultados son adecuados para priorización operativa, pero no sustituyen validación técnica en planta.

## 2. MARCO TEÓRICO

### 2.1 Regresión Logística Binaria

La regresión logística binaria es una técnica de clasificación supervisada utilizada cuando la variable objetivo adopta dos estados mutuamente excluyentes, en este caso **falla** y **no falla**. Su valor en entornos industriales radica en que estima probabilidades, lo cual permite construir esquemas de alertamiento por umbrales de riesgo. A diferencia de una clasificación rígida, este enfoque facilita priorizar activos según criticidad.

Desde la perspectiva gerencial, métricas como **recall** y **AUC-ROC** son particularmente importantes. El recall mide la capacidad de detectar fallas reales; en mantenimiento, omitir una falla puede ser más costoso que generar una alerta preventiva adicional.

### 2.2 Análisis de Clustering (K-Means)

K-Means es una técnica no supervisada que agrupa observaciones similares de acuerdo con su proximidad en el espacio de variables. En operaciones industriales, su utilidad no consiste en “predecir” una falla, sino en identificar **perfiles operativos** que permitan diferenciar máquinas eficientes, intermedias o críticas. De esta forma, la segmentación complementa la clasificación, aportando una visión táctica para intervenciones escalonadas.

El índice **Silhouette** evalúa la separación entre grupos. Un valor bajo, como el observado en este estudio, no invalida el análisis, pero obliga a interpretar los clusters como segmentos gerenciales útiles más que como categorías naturales fuertemente separadas.

### 2.3 Árbol de Decisión

El árbol de decisión es un modelo supervisado altamente apreciado en gestión porque transforma relaciones complejas en reglas comprensibles. Su estructura jerárquica permite contestar preguntas del tipo: **si la vibración es alta y el tiempo sin mantenimiento excede cierto umbral, ¿cuál es el riesgo esperado?**

En proyectos de analítica aplicada, los árboles son valiosos aun cuando su precisión no supere a otros modelos, ya que favorecen la adopción por parte de usuarios no técnicos, supervisores de mantenimiento y jefaturas de planta.

### 2.4 Regresión Lineal Múltiple

La regresión lineal múltiple busca explicar una variable cuantitativa —en este caso, la producción— a partir de múltiples predictores. En manufactura suele emplearse para evaluar sensibilidad y relación entre carga operativa, consumo energético, variables de proceso y output productivo.

Cuando el coeficiente de determinación **R²** es cercano a cero o negativo, se interpreta que el conjunto de variables observadas no explica adecuadamente el fenómeno bajo un supuesto lineal. Este resultado puede ser tan valioso como uno positivo, pues señala la necesidad de ampliar la medición del proceso.

### 2.5 Metodología CRISP-DM

CRISP-DM (**Cross Industry Standard Process for Data Mining**) es una metodología ampliamente utilizada para estructurar proyectos analíticos de manera ordenada y reproducible. Sus fases son:

1. **Entendimiento del negocio**
2. **Entendimiento de los datos**
3. **Preparación de los datos**
4. **Modelado**
5. **Evaluación**
6. **Despliegue**

Su principal aporte en este proyecto es asegurar que el modelado no se quede en resultados estadísticos aislados, sino que se conecte con decisiones de negocio como priorización de mantenimiento, diseño de alertas y seguimiento operativo.

## 3. DESCRIPCIÓN DE LOS DATOS

### 3.1 Características del Dataset

El conjunto de datos está conformado por **1,000 registros operacionales** de **180 máquinas** distribuidas en tres plantas. Cada registro representa una observación del estado operativo de un equipo en un contexto de planta, tipo de máquina y turno.

**Resumen general del dataset**

| Indicador | Valor |
|---|---:|
| Registros | 1,000 |
| Máquinas únicas | 180 |
| Variables totales | 15 |
| Variables categóricas | 5 |
| Variables numéricas | 9 |
| Variable objetivo binaria | 1 |
| Tasa global de falla | 38.5% |
| Plantas | Lima, Arequipa, Trujillo |
| Tipos de máquina | 5 |
| Turnos | Mañana, Tarde, Noche |

### 3.2 Diccionario de Datos

| Variable | Tipo | Descripción | Rango / Categorías |
|---|---|---|---|
| id_registro | Categórica | Identificador único del registro operacional | Ej.: MAN-0001 |
| id_maquina | Categórica | Identificador de la máquina observada | Ej.: MQ-097 |
| planta | Categórica | Sede industrial donde opera la máquina | Lima, Arequipa, Trujillo |
| tipo_maquina | Categórica | Familia tecnológica del activo | Prensa, Torno CNC, Inyectora, Empacadora, Compresor |
| turno | Categórica | Turno de operación | Mañana, Tarde, Noche |
| temperatura_c | Numérica | Temperatura de operación del equipo en °C | 45.0 a 108.9 |
| vibracion_mm_s | Numérica | Vibración medida en mm/s | 0.5 a 9.4 |
| presion_bar | Numérica | Presión de operación en bar | 2.88 a 10.04 |
| horas_operacion | Numérica | Horas acumuladas de operación | 141 a 11,977 |
| antiguedad_anios | Numérica | Antigüedad del equipo en años | 0.2 a 11.0 |
| dias_desde_mantenimiento | Numérica | Días transcurridos desde el último mantenimiento | 1 a 240 |
| produccion_unidades | Numérica | Producción obtenida en unidades | 244 a 1,396 |
| tasa_defectos_pct | Numérica | Porcentaje de defectos del proceso | 0.1% a 9.05% |
| consumo_energia_kwh | Numérica | Consumo energético del equipo en kWh | 247.23 a 950.39 |
| falla_maquina | Binaria | Indicador de falla observada | 0 = no falla, 1 = falla |

### 3.3 Estadísticas Descriptivas

Las estadísticas descriptivas muestran un parque de maquinaria con dispersión operativa relevante. La temperatura promedio fue **73.49 °C**, la vibración promedio **4.26 mm/s** y los días promedio desde mantenimiento **122.16**. La amplitud de rangos sugiere coexistencia de máquinas en condiciones sanas y otras claramente tensionadas.

| Variable | Media | Desv. Est. | Mínimo | Q1 | Mediana | Q3 | Máximo |
|---|---:|---:|---:|---:|---:|---:|---:|
| temperatura_c | 73.49 | 10.04 | 45.00 | 66.50 | 73.50 | 80.12 | 108.90 |
| vibracion_mm_s | 4.26 | 1.69 | 0.50 | 3.13 | 4.26 | 5.42 | 9.40 |
| presion_bar | 6.57 | 1.18 | 2.88 | 5.75 | 6.59 | 7.41 | 10.04 |
| horas_operacion | 6,140.59 | 3,413.22 | 141 | 3,221.50 | 6,324 | 9,113 | 11,977 |
| antiguedad_anios | 4.41 | 2.35 | 0.20 | 2.60 | 4.50 | 6.20 | 11.00 |
| dias_desde_mantenimiento | 122.16 | 68.90 | 1 | 65 | 122 | 182 | 240 |
| tasa_defectos_pct | 3.02 | 1.43 | 0.10 | 2.01 | 3.04 | 3.92 | 9.05 |
| consumo_energia_kwh | 580.75 | 116.55 | 247.23 | 503.80 | 579.07 | 654.26 | 950.39 |
| produccion_unidades | 845.93 | 179.47 | 244 | 718 | 848 | 971 | 1,396 |

### 3.4 Calidad de Datos

La evaluación de calidad evidenció una base robusta para el desarrollo del proyecto:

| Criterio | Resultado |
|---|---:|
| Valores nulos totales | 0 |
| Registros duplicados | 0 |
| Consistencia de categorías | Correcta |
| Variables numéricas fuera de rango | No críticas |
| Preparación adicional requerida | Codificación y escalamiento |

En consecuencia, no fue necesario aplicar imputación, depuración agresiva ni eliminación de registros. Esto mejora la confiabilidad del análisis y reduce sesgos por tratamiento de datos faltantes.

## 4. PREPARACIÓN DE DATOS

### 4.1 Limpieza Realizada

La fase de preparación estuvo orientada principalmente a adecuar el dataset para el modelado. Dado que la calidad estructural era buena, la limpieza se centró en verificar integridad, asegurar tipos de dato consistentes y preparar columnas para consumo de algoritmos.

Las variables categóricas `planta`, `tipo_maquina` y `turno` fueron codificadas mediante **Label Encoding**, generando las columnas `planta_cod`, `tipo_maquina_cod` y `turno_cod`. Este paso permitió integrar información contextual al modelo sin perder trazabilidad del significado original.

### 4.2 Transformaciones Aplicadas

Las principales transformaciones fueron:

- codificación de variables categóricas;
- estandarización de variables predictoras para **regresión logística** y **K-Means**;
- definición de conjuntos de variables específicos según técnica;
- exportación de datasets limpios y artefactos del modelo para reutilización.

Las variables empleadas en los modelos de falla fueron: temperatura, vibración, presión, horas de operación, antigüedad, días desde mantenimiento, tasa de defectos, consumo energético y variables codificadas de planta, tipo de máquina y turno.

### 4.3 Partición Train/Test

Para los modelos de clasificación se utilizó una partición **80% entrenamiento / 20% prueba**, equivalente a **800 registros para entrenamiento** y **200 para prueba**, con **estratificación por la variable `falla_maquina`**. Esta decisión preserva la proporción de fallas en ambos subconjuntos y mejora la validez comparativa de resultados.

En el caso de la regresión lineal múltiple se aplicó igualmente una división **80/20**, sin estratificación, por tratarse de una variable objetivo continua.

## 5. ANÁLISIS EXPLORATORIO

### 5.1 Distribución de la Variable Objetivo

La variable `falla_maquina` presenta **38.5% de eventos de falla** y **61.5% de no falla**. Si bien no se trata de una distribución extremadamente desbalanceada, sí representa un problema de negocio crítico: casi **4 de cada 10 observaciones** reflejan una condición fallida.

Desde la perspectiva gerencial, este nivel de incidencia justifica plenamente una estrategia de mantenimiento predictivo, ya que la frecuencia de ocurrencia excede lo esperable para una operación madura y controlada.

### 5.2 Análisis por Planta y Turno

Las diferencias por planta son moderadas, pero por turno resultan más marcadas.

**Tasa de falla por planta**

| Planta | Registros | Tasa de falla |
|---|---:|---:|
| Lima | 341 | 36.07% |
| Arequipa | 331 | 39.58% |
| Trujillo | 328 | 39.94% |

**Tasa de falla por turno**

| Turno | Registros | Tasa de falla |
|---|---:|---:|
| Mañana | 345 | 32.46% |
| Noche | 319 | 40.44% |
| Tarde | 336 | 42.86% |

El turno **tarde** exhibe la mayor criticidad, con una tasa **10.40 puntos porcentuales** superior a la del turno mañana. Esto constituye un insight de negocio relevante, ya que permite focalizar supervisión, pausas técnicas, inspecciones cortas y dotación de soporte en franjas de mayor exposición.

### 5.3 Análisis por Tipo de Máquina

Las tasas de falla por tipo de máquina son relativamente cercanas entre sí, lo que sugiere que el riesgo no está concentrado exclusivamente en una sola familia tecnológica, sino en la combinación entre condición operativa y mantenimiento.

| Tipo de máquina | Registros | Tasa de falla |
|---|---:|---:|
| Inyectora | 199 | 39.70% |
| Prensa | 210 | 39.05% |
| Empacadora | 215 | 38.14% |
| Compresor | 164 | 37.80% |
| Torno CNC | 212 | 37.74% |

Aunque las diferencias son reducidas, las **inyectoras** presentan la tasa más alta. Además, en el conjunto de alto riesgo, las **empacadoras** concentran **63 registros**, equivalentes al **26.0%** del total priorizado.

### 5.4 Correlaciones Clave

El análisis de correlación con la variable `falla_maquina` muestra que las asociaciones más importantes son:

| Variable | Correlación con falla |
|---|---:|
| temperatura_c | 0.303 |
| vibracion_mm_s | 0.298 |
| dias_desde_mantenimiento | 0.271 |
| consumo_energia_kwh | 0.140 |
| presion_bar | 0.103 |
| tasa_defectos_pct | 0.101 |
| antiguedad_anios | 0.057 |
| horas_operacion | 0.016 |
| produccion_unidades | 0.006 |

Estas correlaciones permiten extraer una lectura importante: el riesgo de falla se asocia más con **señales de condición operativa** y **tiempo sin mantenimiento** que con la producción misma.

### 5.5 Insights Preliminares

Los principales hallazgos exploratorios son los siguientes:

1. La falla de maquinaria es un problema estructural, no marginal: **38.5%** de incidencia.
2. El riesgo aumenta fuera del turno mañana; tarde y noche concentran mayor exposición.
3. Temperatura, vibración y días desde mantenimiento son las señales más informativas.
4. Las diferencias entre plantas existen, pero no son extremas; el problema es transversal.
5. La producción casi no se correlaciona con la falla (**0.006**), señal de que ambas dimensiones deben gestionarse con métricas distintas.

## 6. MODELADO Y RESULTADOS

### 6.1 Modelo 1: Regresión Logística

La regresión logística fue el modelo de mejor rendimiento global para la predicción de falla. Sus resultados fueron:

| Métrica | Resultado |
|---|---:|
| Accuracy | 69.5% |
| Precision | 59.3% |
| Recall | 66.2% |
| F1-Score | 62.6% |
| AUC-ROC | 0.761 |
| CV-5 Accuracy | 73.0% |

La lectura gerencial de estas métricas es positiva. Un **AUC-ROC de 0.761** indica capacidad de discriminación razonable, superior al azar y útil para priorización. El **recall de 66.2%** implica que el modelo identifica alrededor de dos tercios de las fallas reales, lo cual es valioso en un esquema preventivo.

Los coeficientes más influyentes fueron:

| Variable | Efecto relativo en el riesgo |
|---|---:|
| dias_desde_mantenimiento | 0.763 |
| vibracion_mm_s | 0.724 |
| temperatura_c | 0.716 |
| antiguedad_anios | 0.326 |
| presion_bar | 0.277 |

En términos prácticos, a mayor tiempo sin mantenimiento, mayor vibración y mayor temperatura, mayor probabilidad de falla estimada.

### 6.2 Modelo 2: Clustering K-Means

El modelo K-Means se implementó con **k = 4** por interpretabilidad gerencial, obteniéndose un **Silhouette score de 0.082**. Aunque este valor sugiere separación moderada-baja entre clusters, la segmentación resultó útil para diferenciar perfiles de operación y riesgo.

| Segmento | Registros | Falla promedio | Rasgos dominantes |
|---|---:|---:|---|
| Eficientes | 280 | 22.9% | Menor vibración, menor temperatura, consumo controlado |
| Estándar | 275 | 32.0% | Desempeño intermedio, mayor antigüedad y mantenimiento postergado |
| Alerta | 219 | 47.0% | Temperatura y vibración elevadas |
| Críticas | 226 | 57.5% | Máxima temperatura, vibración y consumo energético |

**Perfiles promedio por cluster**

| Segmento | Temperatura °C | Vibración mm/s | Días sin mantenimiento | Antigüedad (años) | Defectos % | Producción | Energía kWh |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eficientes | 68.84 | 3.51 | 110.08 | 2.53 | 2.60 | 850.04 | 510.99 |
| Estándar | 68.81 | 3.58 | 150.61 | 6.13 | 2.51 | 840.52 | 543.08 |
| Alerta | 78.71 | 5.04 | 88.74 | 6.41 | 3.46 | 841.28 | 637.28 |
| Críticas | 79.87 | 5.26 | 134.92 | 2.71 | 3.71 | 851.92 | 658.22 |

Este resultado sugiere que el clustering no define solamente “máquinas buenas o malas”, sino configuraciones operativas con distintos mecanismos de exposición.

### 6.3 Modelo 3: Árbol de Decisión

El árbol de decisión entregó un rendimiento menor en exactitud global, pero mayor facilidad de interpretación.

| Métrica | Resultado |
|---|---:|
| Accuracy | 60.0% |
| Recall | 70.1% |
| AUC-ROC | 0.650 |

Las variables más importantes fueron:

| Variable | Importancia |
|---|---:|
| vibracion_mm_s | 44.9% |
| dias_desde_mantenimiento | 24.9% |
| temperatura_c | 15.9% |
| presion_bar | 9.2% |
| horas_operacion | 2.8% |

Una de las principales reglas obtenidas puede resumirse así:

- si la **vibración es mayor a 5.45 mm/s** y los **días desde mantenimiento superan 52.5**, la probabilidad de clasificar como falla aumenta;
- si la **vibración es menor o igual a 5.45 mm/s**, pero los **días sin mantenimiento superan 165.5** y la temperatura es elevada, el riesgo también se incrementa.

Estas reglas son particularmente útiles para protocolos de inspección y checklist de mantenimiento.

### 6.4 Modelo 4: Regresión Lineal Múltiple

El modelo de regresión lineal sobre `produccion_unidades` arrojó los siguientes resultados:

| Métrica | Resultado |
|---|---:|
| R² | -0.009 |
| RMSE | 181 unidades |

El valor negativo de **R²** indica que el modelo lineal no logra explicar la producción mejor que una predicción basada simplemente en el promedio. Esto no significa que el análisis haya fallado; significa que la producción depende de factores adicionales no incorporados en el dataset.

Entre esos factores plausibles se encuentran:

- habilidad y experiencia del operador;
- calidad y homogeneidad de la materia prima;
- mix de productos fabricados;
- tiempos de setup y cambio de lote;
- programación y secuenciación de órdenes;
- disponibilidad de insumos;
- microparadas no registradas;
- disciplina operativa y condiciones ambientales.

Por tanto, el bajo R² es un hallazgo de negocio: **la productividad no debe ser inferida solo a partir de señales de condición mecánica**.

### 6.5 Comparación de Técnicas

| Técnica | Objetivo | Fortaleza principal | Limitación principal | Uso recomendado |
|---|---|---|---|---|
| Regresión Logística | Predecir falla | Mejor balance global de métricas | Interpretabilidad media | Sistema principal de riesgo |
| K-Means | Segmentar máquinas | Clasificación operativa gerencial | Silhouette bajo | Priorización táctica |
| Árbol de Decisión | Explicar falla | Reglas claras y accionables | Menor exactitud | Protocolos y umbrales |
| Regresión Lineal | Explicar producción | Evidencia sobre variables faltantes | Muy bajo ajuste | Rediseñar captura de datos |

En síntesis, la mejor arquitectura analítica no es escoger un solo modelo, sino combinar:

- **regresión logística** para scoring predictivo;
- **árbol de decisión** para reglas operativas;
- **clustering** para segmentación de mantenimiento;
- **regresión lineal** como diagnóstico de variables faltantes en productividad.

## 7. INTERPRETACIÓN GERENCIAL

### 7.1 Traducción de Métricas a Lenguaje de Negocio

Para un tomador de decisiones, la interpretación de métricas debe centrarse en impacto operativo:

- **69.5% de accuracy** significa que el modelo acierta aproximadamente 7 de cada 10 casos.
- **66.2% de recall** implica detectar cerca de 66 de cada 100 fallas reales antes de que pasen inadvertidas.
- **AUC-ROC de 0.761** refleja una capacidad consistente para diferenciar activos más riesgosos de activos más estables.
- **242 registros de alto riesgo** constituyen una cartera inmediata de intervención.

En otras palabras, el modelo no reemplaza al especialista de mantenimiento, pero sí le permite trabajar con una cola priorizada, reduciendo inspecciones ciegas.

### 7.2 Máquinas Priorizadas para Mantenimiento

El análisis identificó **242 observaciones con probabilidad de falla superior a 70%**, asociadas a **137 máquinas únicas**. La distribución de estos casos es transversal entre plantas, lo que confirma que el riesgo no se concentra en una sola sede:

| Planta | Registros de alto riesgo | Participación |
|---|---:|---:|
| Lima | 81 | 33.5% |
| Trujillo | 81 | 33.5% |
| Arequipa | 80 | 33.1% |

Los casos de mayor riesgo muestran patrones repetidos: temperaturas superiores a 85 °C, vibraciones por encima de 6 mm/s y periodos extensos sin mantenimiento. Por tanto, la recomendación no es revisar aleatoriamente todas las máquinas, sino intervenir primero aquellas con acumulación simultánea de señales críticas.

### 7.3 Segmentos Operativos y Plan de Acción

La segmentación permite diferenciar estrategias:

| Segmento | Riesgo | Acción sugerida |
|---|---|---|
| Eficientes | Bajo | Mantener frecuencia actual y monitoreo rutinario |
| Estándar | Medio-bajo | Revisar antigüedad, limpieza y mantenimiento calendarizado |
| Alerta | Medio-alto | Inspección preventiva priorizada en 7 días |
| Críticas | Alto | Intervención inmediata y evaluación de parada programada |

Además, entre los registros de alto riesgo, el **43.4%** pertenece al segmento **Críticas** y el **31.4%** al segmento **Alerta**. Esto confirma la coherencia entre el clustering y el scoring predictivo.

### 7.4 Factores Críticos de Eficiencia

Los factores que más afectan la estabilidad operativa son:

1. **Vibración**: variable más importante del árbol (**44.9%**).
2. **Días desde mantenimiento**: principal coeficiente logístico (**0.763**).
3. **Temperatura**: mayor correlación con falla (**0.303**).
4. **Turno**: el turno tarde eleva la exposición a falla.
5. **Consumo energético**: mayor en segmentos críticos (**658.22 kWh**) frente a eficientes (**510.99 kWh**).

Cinco insights de negocio respaldados por números son, por tanto:

1. La operación presenta una falla estructural de **38.5%**.
2. El turno tarde alcanza **42.86%** de falla, superando claramente al turno mañana (**32.46%**).
3. Los segmentos **Alerta** y **Críticas** concentran tasas de falla de **47.0%** y **57.5%**, respectivamente.
4. La vibración explica **44.9%** de la importancia del árbol de decisión.
5. Existen **242 registros de alto riesgo** que permiten actuar selectivamente en lugar de intervenir indiscriminadamente.

## 8. SISTEMA DE ALERTAS Y PRIORIZACIÓN

### 8.1 Umbrales de Riesgo

Se propone un esquema de alertas basado en la probabilidad estimada por la regresión logística y reforzado con reglas del árbol de decisión.

| Nivel | Probabilidad de falla | Condición operativa sugerida | Prioridad |
|---|---:|---|---|
| Verde | < 40% | Operación estable | Baja |
| Amarillo | 40% a < 70% | Riesgo moderado | Media |
| Rojo | ≥ 70% | Alto riesgo de falla | Alta |

Umbrales complementarios recomendados:

- **vibración > 5.45 mm/s**;
- **días desde mantenimiento > 165**;
- **temperatura > 79 °C** en contextos de vibración elevada.

### 8.2 Matriz de Priorización

| Probabilidad / Condición | Baja severidad | Media severidad | Alta severidad |
|---|---|---|---|
| < 40% | Monitoreo rutinario | Monitoreo semanal | Inspección preventiva |
| 40% a < 70% | Revisión programada | Inspección en 72 horas | Intervención en 24-48 horas |
| ≥ 70% | Inspección inmediata | Orden de trabajo urgente | Posible parada programada |

La severidad debe considerar criticidad del equipo, impacto en seguridad, afectación sobre calidad y dependencia del proceso productivo.

### 8.3 Reglas de Acción Recomendadas

1. Generar orden automática de inspección cuando la probabilidad de falla sea **≥ 70%**.
2. Escalar a mantenimiento urgente si vibración **> 5.45 mm/s** y temperatura **> 79 °C**.
3. Programar mantenimiento si los días sin intervención superan **165 días**, incluso con vibración moderada.
4. Incrementar supervisión en turno tarde y noche por sus tasas de falla de **42.86%** y **40.44%**.
5. Usar el segmento del cluster como criterio adicional: **Críticas** y **Alerta** primero.
6. Registrar causa raíz posterior a cada intervención para retroalimentar el modelo.

## 9. CONCLUSIONES

1. **La tasa de falla de 38.5% confirma un problema operativo de alta prioridad**, ya que 385 de 1,000 registros evidencian fallas de maquinaria, nivel incompatible con una estrategia predominantemente correctiva.
2. **La regresión logística fue la técnica supervisada más efectiva**, con Accuracy de 69.5%, Recall de 66.2% y AUC-ROC de 0.761, lo que la convierte en la mejor base para un sistema de alertas predictivas.
3. **La vibración, la temperatura y los días desde mantenimiento son los principales impulsores del riesgo**, sustentado por correlaciones de 0.298, 0.303 y 0.271, respectivamente, y por la importancia del árbol de decisión, donde la vibración explica 44.9%.
4. **El turno influye materialmente en la exposición al fallo**, puesto que la tasa de falla sube de 32.46% en la mañana a 42.86% en la tarde, diferencia de 10.40 puntos porcentuales que debe ser gestionada operativamente.
5. **La segmentación K-Means permitió construir una tipología útil de mantenimiento**, diferenciando grupos Eficientes (22.9% de falla), Estándar (32.0%), Alerta (47.0%) y Críticas (57.5%), aun cuando el índice Silhouette fue bajo (0.082).
6. **La identificación de 242 registros de alto riesgo, asociados a 137 máquinas únicas, constituye el principal entregable táctico del proyecto**, porque permite focalizar recursos de mantenimiento donde la probabilidad de falla supera 70%.
7. **La producción no se explica linealmente con las variables observadas**, como evidencia el R² de -0.009, por lo que la empresa debe ampliar su captura de datos hacia factores humanos, logísticos y de proceso si desea modelar productividad con mayor rigor.

## 10. RECOMENDACIONES ESTRATÉGICAS

### 10.1 Acciones Inmediatas (Corto Plazo — 0 a 3 meses)

1. **Intervenir de forma prioritaria los 242 registros de alto riesgo**, comenzando por los casos del segmento **Críticas**.
2. **Implementar un tablero de alertas en Streamlit** con semáforos por nivel de riesgo, planta, turno y tipo de máquina.
3. **Revisar inmediatamente equipos con vibración superior a 5.45 mm/s**, aunque no hayan fallado aún.
4. **Programar mantenimiento correctivo-preventivo para equipos con más de 165 días sin intervención**.
5. **Reforzar supervisión en turno tarde**, incorporando checklists rápidos de temperatura, vibración y presión al inicio y mitad del turno.

### 10.2 Acciones Tácticas (Mediano Plazo — 3 a 12 meses)

6. **Rediseñar la política de mantenimiento preventivo** usando probabilidad de falla y no solo calendario fijo.
7. **Crear indicadores semanales por planta y turno**: tasa de falla, promedio de vibración, días desde mantenimiento y porcentaje de activos en rojo.
8. **Capacitar a supervisores y técnicos en lectura de alertas analíticas**, para reducir la brecha entre resultado del modelo y acción en planta.
9. **Registrar causa raíz, repuestos usados y tiempo de parada por intervención**, enriqueciendo futuros modelos de confiabilidad.
10. **Revisar específicamente empacadoras e inyectoras**, por su peso relativo en los casos priorizados y sus tasas de falla cercanas o superiores al promedio global.

### 10.3 Acciones Estratégicas (Largo Plazo — 1 a 3 años)

11. **Evolucionar hacia mantenimiento predictivo con series temporales e IoT**, capturando señales continuas de sensores y eventos.
12. **Ampliar el modelo de producción** incorporando variables de operador, calidad de materia prima, mix de producto, setups, disponibilidad de insumos y microparadas.
13. **Integrar analítica predictiva con el sistema de gestión de mantenimiento (CMMS/ERP)** para que las alertas generen automáticamente órdenes de trabajo y trazabilidad económica.

## 11. REFERENCIAS

1. Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS.
2. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
4. Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
5. Montgomery, D. C. (2019). *Introduction to Statistical Quality Control* (8th ed.). Wiley.
6. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O’Reilly.
7. ISO 55000. (2014). *Asset management — Overview, principles and terminology*. International Organization for Standardization.

## ANEXOS

### Anexo A: Código Python Resumen

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("data/manufactura/dataset_manufactura_1000.csv")

for col in ["planta", "tipo_maquina", "turno"]:
    df[f"{col}_cod"] = LabelEncoder().fit_transform(df[col])

vars_modelo = [
    "temperatura_c", "vibracion_mm_s", "presion_bar", "horas_operacion",
    "antiguedad_anios", "dias_desde_mantenimiento", "tasa_defectos_pct",
    "consumo_energia_kwh", "planta_cod", "tipo_maquina_cod", "turno_cod"
]

X = df[vars_modelo]
y = df["falla_maquina"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_train_sc, y_train)
y_prob = modelo_log.predict_proba(X_test_sc)[:, 1]
y_pred = modelo_log.predict(X_test_sc)

modelo_tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
modelo_tree.fit(X_train, y_train)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(StandardScaler().fit_transform(X))

modelo_reg = LinearRegression()
modelo_reg.fit(
    df[["temperatura_c", "vibracion_mm_s", "presion_bar", "horas_operacion",
        "consumo_energia_kwh", "dias_desde_mantenimiento", "tasa_defectos_pct"]],
    df["produccion_unidades"]
)
```

### Anexo B: Glosario de Términos

| Término | Definición |
|---|---|
| Accuracy | Proporción total de predicciones correctas del modelo. |
| Precision | Porcentaje de alertas positivas que realmente corresponden a fallas. |
| Recall | Porcentaje de fallas reales correctamente detectadas. |
| F1-Score | Media armónica entre precision y recall. |
| AUC-ROC | Medida de capacidad discriminatoria del clasificador. |
| Clustering | Técnica de segmentación no supervisada para agrupar observaciones similares. |
| Silhouette Score | Indicador de separación y cohesión de clusters. |
| Feature Importance | Peso relativo de una variable dentro de un modelo como árbol de decisión. |
| Mantenimiento predictivo | Enfoque que anticipa fallas con base en datos y señales operativas. |
| Mantenimiento preventivo | Intervención programada antes de la ocurrencia de una falla. |
| RMSE | Error cuadrático medio; mide la magnitud del error en una regresión. |
| R² | Porcentaje de variabilidad explicada por un modelo de regresión. |
| CRISP-DM | Metodología estándar para proyectos de minería de datos y analítica. |
| Variable objetivo | Resultado que se desea explicar o predecir mediante un modelo. |
| Umbral de riesgo | Punto de corte usado para activar una alerta o intervención. |
