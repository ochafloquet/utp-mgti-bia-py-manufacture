"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       ANÁLISIS DE MANUFACTURA - PROYECTO FINAL UTP MGTI BIA                ║
║       Business Intelligence & Analytics - Sesión 60271                     ║
║       Universidad Tecnológica del Perú                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Técnicas aplicadas:
  1. Regresión Logística Binaria   → Predicción de fallas de maquinaria
  2. Clustering K-Means            → Segmentación operativa de máquinas
  3. Árbol de Decisión             → Reglas interpretables de falla
  4. Regresión Lineal Múltiple     → Factores que impactan producción
  5. Análisis de Correlación       → Relaciones entre variables

Ejecutar:
    python scripts/analisis_manufactura.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: CONFIGURACIÓN E IMPORTACIONES
# ══════════════════════════════════════════════════════════════════════════════

import warnings
import sys
import io
from pathlib import Path

# Forzar salida UTF-8 en Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score, r2_score, mean_squared_error, mean_absolute_error,
    ConfusionMatrixDisplay
)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'manufactura'
RESULTADOS_DIR = BASE_DIR / 'resultados' / 'manufactura'
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

SEPARADOR = "=" * 80
SEP_SEC   = "-" * 60

print(SEPARADOR)
print("  ANÁLISIS DE MANUFACTURA — PROYECTO FINAL UTP MGTI BIA")
print(SEPARADOR)
print(f"  Directorio de datos   : {DATA_DIR}")
print(f"  Directorio resultados : {RESULTADOS_DIR}")
print(SEPARADOR)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/13] Cargando datos...")

csv_path = DATA_DIR / 'dataset_manufactura_1000.csv'
if not csv_path.exists():
    print(f"  ⚠  Archivo no encontrado: {csv_path}")
    print("  ➜  Generando dataset sintético de ejemplo...")
    gen_script = DATA_DIR / 'generar_dataset.py'
    if gen_script.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("gen", gen_script)
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        gen.generar_dataset()
    else:
        print("  ✗  No se encontró el generador. Coloca el CSV en data/manufactura/")
        sys.exit(1)

df = pd.read_csv(csv_path)
print(f"  ✓ Dataset cargado: {df.shape[0]:,} registros × {df.shape[1]} variables")
print(f"  ✓ Columnas: {list(df.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: EXPLORACIÓN INICIAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/13] Exploración inicial del dataset...")
print(SEP_SEC)

print("\n--- Tipos de datos e información general ---")
df.info()

print("\n--- Primeras 5 filas ---")
print(df.head().to_string())

print("\n--- Estadísticas descriptivas (variables numéricas) ---")
print(df.describe().round(2).to_string())

print("\n--- Valores nulos por columna ---")
nulos = df.isnull().sum()
if nulos.sum() == 0:
    print("  ✓ Sin valores nulos")
else:
    print(nulos[nulos > 0])

print("\n--- Duplicados ---")
dupl = df.duplicated().sum()
print(f"  Registros duplicados: {dupl}")

print("\n--- Variable objetivo: falla_maquina ---")
print(df['falla_maquina'].value_counts().to_string())
tasa_falla = df['falla_maquina'].mean() * 100
print(f"  Tasa de falla global: {tasa_falla:.2f}%")

print("\n--- Distribución por variables categóricas ---")
for col in ['planta', 'tipo_maquina', 'turno']:
    print(f"\n  {col}:")
    print(df.groupby(col)['falla_maquina'].agg(['count', 'mean'])
            .rename(columns={'count': 'total', 'mean': 'tasa_falla'})
            .assign(tasa_falla=lambda x: (x['tasa_falla'] * 100).round(1))
            .to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: LIMPIEZA Y TRANSFORMACIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/13] Limpieza y transformación de datos...")

df_clean = df.copy()

# Encoding de variables categóricas
le_dict = {}
categoricas = ['planta', 'tipo_maquina', 'turno']
for col in categoricas:
    le = LabelEncoder()
    df_clean[f'{col}_cod'] = le.fit_transform(df_clean[col])
    le_dict[col] = le
    joblib.dump(le, RESULTADOS_DIR / f'encoder_{col}.pkl')
    print(f"  ✓ '{col}' codificado → '{col}_cod'  clases: {list(le.classes_)}")

# Guardar dataset limpio
df_clean.to_csv(RESULTADOS_DIR / 'dataset_limpio.csv', index=False)
print(f"\n  ✓ Dataset limpio guardado ({df_clean.shape[0]} filas × {df_clean.shape[1]} cols)")

# Definir conjuntos de variables para los modelos
VARS_NUMERICAS = ['temperatura_c', 'vibracion_mm_s', 'presion_bar',
                  'horas_operacion', 'antiguedad_anios',
                  'dias_desde_mantenimiento', 'tasa_defectos_pct',
                  'consumo_energia_kwh', 'produccion_unidades']

VARS_MODELO = ['temperatura_c', 'vibracion_mm_s', 'presion_bar',
               'horas_operacion', 'antiguedad_anios',
               'dias_desde_mantenimiento', 'tasa_defectos_pct',
               'consumo_energia_kwh', 'planta_cod', 'tipo_maquina_cod', 'turno_cod']

VARS_REGRESION = ['temperatura_c', 'vibracion_mm_s', 'presion_bar',
                  'horas_operacion', 'consumo_energia_kwh',
                  'dias_desde_mantenimiento', 'tasa_defectos_pct']


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: ANÁLISIS EXPLORATORIO — VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/13] Generando visualizaciones exploratorias...")

# ── VIZ 01: Distribución de la variable objetivo ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribución de Fallas de Maquinaria', fontsize=14, fontweight='bold')

colores = ['#2ecc71', '#e74c3c']
conteos = df['falla_maquina'].value_counts().sort_index()
labels = ['Sin Falla (0)', 'Con Falla (1)']

axes[0].bar(labels, conteos.values, color=colores, edgecolor='white', linewidth=1.5)
axes[0].set_ylabel('Número de Registros')
axes[0].set_title('Conteo absoluto')
for i, v in enumerate(conteos.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].pie(conteos.values, labels=labels, colors=colores, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Proporción')

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_01_distribucion_fallas.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_01_distribucion_fallas.png")

# ── VIZ 02: Histogramas de variables numéricas ────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Distribuciones de Variables Numéricas', fontsize=14, fontweight='bold')

for idx, col in enumerate(VARS_NUMERICAS):
    ax = axes[idx // 3, idx % 3]
    df[col].hist(bins=30, ax=ax, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Frecuencia')
    media = df[col].mean()
    ax.axvline(media, color='red', linestyle='--', linewidth=1.5, label=f'μ={media:.1f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_02_histogramas.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_02_histogramas.png")

# ── VIZ 03: Boxplots por tipo de máquina ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Variables Operativas por Tipo de Máquina', fontsize=14, fontweight='bold')

vars_box = ['temperatura_c', 'vibracion_mm_s', 'presion_bar', 'tasa_defectos_pct']
for idx, col in enumerate(vars_box):
    ax = axes[idx // 2, idx % 2]
    df.boxplot(column=col, by='tipo_maquina', ax=ax, vert=True,
               patch_artist=True, medianprops=dict(color='red', linewidth=2))
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Tipo de Máquina')
    plt.sca(ax)
    plt.xticks(rotation=30, ha='right')

fig.suptitle('Variables Operativas por Tipo de Máquina', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_03_boxplots_tipo_maquina.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_03_boxplots_tipo_maquina.png")

# ── VIZ 04: Matriz de correlación ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
cols_corr = VARS_NUMERICAS + ['falla_maquina']
correlaciones = df[cols_corr].corr()
mask = np.triu(np.ones_like(correlaciones, dtype=bool))
sns.heatmap(correlaciones, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Matriz de Correlación — Variables Numéricas', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_04_correlacion.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_04_correlacion.png")

# ── VIZ 05: Fallas por planta y turno ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Tasa de Fallas por Planta y Turno', fontsize=14, fontweight='bold')

fallas_planta = df.groupby('planta')['falla_maquina'].mean() * 100
fallas_turno  = df.groupby('turno')['falla_maquina'].mean() * 100

fallas_planta.sort_values(ascending=False).plot(
    kind='bar', ax=axes[0], color='#e74c3c', edgecolor='white')
axes[0].set_title('Por Planta')
axes[0].set_ylabel('Tasa de Falla (%)')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(fallas_planta.sort_values(ascending=False)):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

fallas_turno.sort_values(ascending=False).plot(
    kind='bar', ax=axes[1], color='#e67e22', edgecolor='white')
axes[1].set_title('Por Turno')
axes[1].set_ylabel('Tasa de Falla (%)')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=0)
for i, v in enumerate(fallas_turno.sort_values(ascending=False)):
    axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_05_fallas_planta_turno.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_05_fallas_planta_turno.png")

# ── VIZ 06: Violinplots falla vs variables clave ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribución de Variables según Estado de Falla', fontsize=14, fontweight='bold')

vars_violin = ['temperatura_c', 'vibracion_mm_s', 'dias_desde_mantenimiento',
               'antiguedad_anios', 'tasa_defectos_pct', 'consumo_energia_kwh']
for idx, col in enumerate(vars_violin):
    ax = axes[idx // 3, idx % 3]
    plot_df = df.copy()
    plot_df['Falla'] = plot_df['falla_maquina'].map({0: 'Sin Falla', 1: 'Con Falla'})
    sns.violinplot(data=plot_df, x='Falla', y=col, ax=ax,
                   palette={'Sin Falla': '#2ecc71', 'Con Falla': '#e74c3c'}, inner='quartile')
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_06_violin_falla.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_06_violin_falla.png")

# ── VIZ 07: Scatter temperatura vs vibración (coloreado por falla) ────────
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df['temperatura_c'], df['vibracion_mm_s'],
                     c=df['falla_maquina'], cmap='RdYlGn_r',
                     alpha=0.6, s=25, edgecolors='none')
plt.colorbar(scatter, ax=ax, label='Falla (0=No, 1=Sí)')
ax.set_xlabel('Temperatura (°C)', fontsize=11)
ax.set_ylabel('Vibración (mm/s)', fontsize=11)
ax.set_title('Temperatura vs Vibración — Estado de Falla', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_07_scatter_temp_vibracion.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_07_scatter_temp_vibracion.png")

# ── VIZ 08: Producción y defectos por tipo de máquina ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Eficiencia Operativa por Tipo de Máquina', fontsize=14, fontweight='bold')

prod_tipo = df.groupby('tipo_maquina')['produccion_unidades'].mean().sort_values(ascending=False)
def_tipo  = df.groupby('tipo_maquina')['tasa_defectos_pct'].mean().sort_values(ascending=False)

prod_tipo.plot(kind='bar', ax=axes[0], color='#27ae60', edgecolor='white')
axes[0].set_title('Producción Media por Tipo')
axes[0].set_ylabel('Unidades')
axes[0].tick_params(axis='x', rotation=30)

def_tipo.plot(kind='bar', ax=axes[1], color='#c0392b', edgecolor='white')
axes[1].set_title('Tasa de Defectos Media por Tipo')
axes[1].set_ylabel('Defectos (%)')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'viz_08_eficiencia_tipo.png', bbox_inches='tight')
plt.close()
print("  ✓ viz_08_eficiencia_tipo.png")

print(f"\n  ✓ 8 visualizaciones exploratorias generadas en {RESULTADOS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: DIVISIÓN TRAIN / TEST
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5/13] División Train/Test (80/20, stratified)...")

X = df_clean[VARS_MODELO]
y = df_clean['falla_maquina']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test:  {X_test.shape[0]}  registros ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"  Tasa de falla — Train: {y_train.mean()*100:.1f}%  |  Test: {y_test.mean()*100:.1f}%")

# Escalar para regresión logística y clustering
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, RESULTADOS_DIR / 'scaler.pkl')
print("  ✓ StandardScaler guardado")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: MODELO 1 — REGRESIÓN LOGÍSTICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6/13] Entrenando Modelo 1: Regresión Logística...")
print(SEP_SEC)

lr_model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_sc, y_train)

y_pred_lr   = lr_model.predict(X_test_sc)
y_proba_lr  = lr_model.predict_proba(X_test_sc)[:, 1]

acc_lr  = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
rec_lr  = recall_score(y_test, y_pred_lr, zero_division=0)
f1_lr   = f1_score(y_test, y_pred_lr, zero_division=0)

# Validación cruzada (5-fold)
cv_scores_lr = cross_val_score(lr_model, X_train_sc, y_train, cv=5, scoring='accuracy')

print(f"  Accuracy   : {acc_lr:.4f}")
print(f"  Precision  : {prec_lr:.4f}")
print(f"  Recall     : {rec_lr:.4f}")
print(f"  F1-Score   : {f1_lr:.4f}")
print(f"  CV-5 Acc   : {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
print("\n  Reporte de clasificación:")
print(classification_report(y_test, y_pred_lr,
      target_names=['Sin Falla', 'Con Falla'], zero_division=0))

# Curva ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
print(f"  AUC-ROC    : {roc_auc_lr:.4f}")

# ── Gráficos del modelo logístico ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Regresión Logística — Resultados', fontsize=14, fontweight='bold')

# Matriz de confusión
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(cm_lr, display_labels=['Sin Falla', 'Con Falla'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Matriz de Confusión\nAccuracy={acc_lr:.3f}', fontweight='bold')

# Curva ROC
axes[1].plot(fpr_lr, tpr_lr, color='#2980b9', lw=2.5,
             label=f'ROC (AUC = {roc_auc_lr:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar')
axes[1].fill_between(fpr_lr, tpr_lr, alpha=0.1, color='#2980b9')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Curva ROC', fontweight='bold')
axes[1].legend()

# Coeficientes (Top 10)
coef_df = pd.DataFrame({
    'variable': VARS_MODELO,
    'coeficiente': lr_model.coef_[0]
}).sort_values('coeficiente', key=abs, ascending=True).tail(10)
colores_coef = ['#e74c3c' if v > 0 else '#27ae60' for v in coef_df['coeficiente']]
axes[2].barh(coef_df['variable'], coef_df['coeficiente'], color=colores_coef)
axes[2].axvline(0, color='black', linewidth=0.8)
axes[2].set_title('Top 10 Coeficientes', fontweight='bold')
axes[2].set_xlabel('Coeficiente')

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'modelo_01_logistica.png', bbox_inches='tight')
plt.close()
print("  ✓ Gráfico modelo_01_logistica.png guardado")

joblib.dump(lr_model, RESULTADOS_DIR / 'modelo_logistic_regression.pkl')
print("  ✓ Modelo guardado: modelo_logistic_regression.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: MODELO 2 — K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[7/13] Entrenando Modelo 2: K-Means Clustering...")
print(SEP_SEC)

X_cluster = df_clean[VARS_MODELO].copy()
scaler_clust = StandardScaler()
X_scaled = scaler_clust.fit_transform(X_cluster)

# Método del codo + Silhouette
inertias  = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

k_optimo = K_range[silhouettes.index(max(silhouettes))]
print(f"  K óptimo (mayor Silhouette): {k_optimo}")
print(f"  Silhouette score: {max(silhouettes):.4f}")

# ── Gráfico del codo + silhouette ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means: Selección del Número Óptimo de Clusters', fontsize=13, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'bo-', markersize=7, linewidth=2)
axes[0].set_xlabel('Número de clusters (k)')
axes[0].set_ylabel('Inercia (WSS)')
axes[0].set_title('Método del Codo')
axes[0].axvline(k_optimo, color='red', linestyle='--', label=f'k={k_optimo}')
axes[0].legend()

axes[1].plot(list(K_range), silhouettes, 'rs-', markersize=7, linewidth=2)
axes[1].set_xlabel('Número de clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por k')
axes[1].axvline(k_optimo, color='red', linestyle='--', label=f'k={k_optimo}')
axes[1].legend()

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'modelo_02_elbow_kmeans.png', bbox_inches='tight')
plt.close()
print("  ✓ modelo_02_elbow_kmeans.png guardado")

# Clustering final
k_final = 4  # usar 4 para interpretabilidad de negocio
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)
df_clean['cluster'] = clusters

silhouette_final = silhouette_score(X_scaled, clusters)
print(f"  Clusters finales (k=4): Silhouette = {silhouette_final:.4f}")

# Perfiles de clusters
perfil_cols = ['temperatura_c', 'vibracion_mm_s', 'dias_desde_mantenimiento',
               'antiguedad_anios', 'tasa_defectos_pct', 'falla_maquina',
               'produccion_unidades', 'consumo_energia_kwh']
perfiles = df_clean.groupby('cluster')[perfil_cols].mean().round(2)
print("\n  Perfiles de clusters:")
print(perfiles.to_string())
perfiles.to_csv(RESULTADOS_DIR / 'segmentos_clustering.csv')

# Nombrar clusters según perfil de riesgo
perfil_riesgo = perfiles['falla_maquina'].sort_values()
nombres_cluster = {}
for i, (c, _) in enumerate(perfil_riesgo.items()):
    etiquetas = ['Eficientes', 'Estándar', 'Alerta', 'Críticas']
    nombres_cluster[c] = etiquetas[i]
print(f"\n  Etiquetas asignadas: {nombres_cluster}")

df_clean['cluster_nombre'] = df_clean['cluster'].map(nombres_cluster)

# ── Visualización PCA 2D de clusters ──────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'K-Means Clustering (k={k_final}) — Visualización PCA',
             fontsize=13, fontweight='bold')

colores_cluster = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
for c in range(k_final):
    mask_c = clusters == c
    axes[0].scatter(X_pca[mask_c, 0], X_pca[mask_c, 1],
                    c=colores_cluster[c], label=f'Cluster {c}: {nombres_cluster[c]}',
                    alpha=0.6, s=20)
axes[0].set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% varianza)')
axes[0].set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% varianza)')
axes[0].set_title('Clusters en espacio PCA')
axes[0].legend(markerscale=2)

# Heatmap de perfiles de cluster
sns.heatmap(perfiles[perfil_cols[:6]].T, annot=True, fmt='.2f',
            cmap='YlOrRd', ax=axes[1],
            xticklabels=[f"C{c}\n{nombres_cluster[c]}" for c in perfiles.index])
axes[1].set_title('Perfil de Variables por Cluster')

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'modelo_02_clusters_pca.png', bbox_inches='tight')
plt.close()
print("  ✓ modelo_02_clusters_pca.png guardado")

joblib.dump(kmeans_final, RESULTADOS_DIR / 'modelo_kmeans.pkl')
joblib.dump(scaler_clust, RESULTADOS_DIR / 'scaler_cluster.pkl')
print("  ✓ Modelo guardado: modelo_kmeans.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: MODELO 3 — ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n[8/13] Entrenando Modelo 3: Árbol de Decisión...")
print(SEP_SEC)

dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=30,
    min_samples_leaf=15,
    class_weight='balanced',
    random_state=42
)
dt_model.fit(X_train, y_train)

y_pred_dt   = dt_model.predict(X_test)
y_proba_dt  = dt_model.predict_proba(X_test)[:, 1]

acc_dt  = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, zero_division=0)
rec_dt  = recall_score(y_test, y_pred_dt, zero_division=0)
f1_dt   = f1_score(y_test, y_pred_dt, zero_division=0)

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

cv_scores_dt = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"  Accuracy   : {acc_dt:.4f}")
print(f"  Precision  : {prec_dt:.4f}")
print(f"  Recall     : {rec_dt:.4f}")
print(f"  F1-Score   : {f1_dt:.4f}")
print(f"  AUC-ROC    : {roc_auc_dt:.4f}")
print(f"  CV-5 Acc   : {cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}")

# Importancias
importancias = pd.DataFrame({
    'variable': VARS_MODELO,
    'importancia': dt_model.feature_importances_
}).sort_values('importancia', ascending=False)
print("\n  Feature Importances (Top 5):")
print(importancias.head().to_string(index=False))

# Reglas en texto
reglas_texto = export_text(dt_model, feature_names=VARS_MODELO)
reglas_path = RESULTADOS_DIR / 'reglas_decision.txt'
with open(reglas_path, 'w', encoding='utf-8') as f:
    f.write("REGLAS DEL ÁRBOL DE DECISIÓN — MANUFACTURA UTP\n")
    f.write("=" * 60 + "\n\n")
    f.write(reglas_texto)
print(f"  ✓ Reglas guardadas en: {reglas_path.name}")

# ── Gráficos del árbol ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Árbol de Decisión — Resultados', fontsize=14, fontweight='bold')

# Feature importances
imp_plot = importancias.set_index('variable')['importancia'].sort_values()
colores_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp_plot)))
imp_plot.plot(kind='barh', ax=axes[0], color=colores_imp)
axes[0].set_title('Feature Importances', fontweight='bold')
axes[0].set_xlabel('Importancia')
for i, v in enumerate(imp_plot):
    axes[0].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

# Curva ROC
axes[1].plot(fpr_dt, tpr_dt, color='#8e44ad', lw=2.5,
             label=f'Árbol (AUC={roc_auc_dt:.3f})')
axes[1].plot(fpr_lr, tpr_lr, color='#2980b9', lw=1.5, linestyle='--',
             label=f'Logística (AUC={roc_auc_lr:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Curva ROC — Comparación', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'modelo_03_arbol_resultados.png', bbox_inches='tight')
plt.close()
print("  ✓ modelo_03_arbol_resultados.png guardado")

# Árbol visual (reducido, profundidad 3 para legibilidad)
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(dt_model, max_depth=3, filled=True,
          feature_names=VARS_MODELO,
          class_names=['Sin Falla', 'Con Falla'],
          ax=ax, fontsize=8, impurity=True,
          proportion=False, rounded=True)
ax.set_title('Árbol de Decisión (máx. profundidad=3 para visualización)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'modelo_03_arbol_visual.png', bbox_inches='tight', dpi=200)
plt.close()
print("  ✓ modelo_03_arbol_visual.png guardado")

joblib.dump(dt_model, RESULTADOS_DIR / 'modelo_decision_tree.pkl')
print("  ✓ Modelo guardado: modelo_decision_tree.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: MODELO 4 — REGRESIÓN LINEAL MÚLTIPLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n[9/13] Entrenando Modelo 4: Regresión Lineal Múltiple...")
print(SEP_SEC)

X_reg = df_clean[VARS_REGRESION].copy()
y_reg = df_clean['produccion_unidades']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lreg_model = LinearRegression()
lreg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = lreg_model.predict(X_reg_test)

r2   = r2_score(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
mae  = mean_absolute_error(y_reg_test, y_reg_pred)

print(f"  R² Score : {r2:.4f}")
print(f"  RMSE     : {rmse:.2f} unidades")
print(f"  MAE      : {mae:.2f} unidades")

# Statsmodels para p-values y estadísticas
X_sm = sm.add_constant(X_reg_train)
modelo_sm = sm.OLS(y_reg_train, X_sm).fit()
print("\n  Resumen OLS (statsmodels):")
print(modelo_sm.summary().as_text())

# VIF — multicolinealidad
vif_data = pd.DataFrame()
vif_data['variable'] = VARS_REGRESION
vif_data['VIF'] = [
    variance_inflation_factor(X_reg.values, i)
    for i in range(len(VARS_REGRESION))
]
print("\n  VIF (multicolinealidad):")
print(vif_data.sort_values('VIF', ascending=False).to_string(index=False))
vif_data.to_csv(RESULTADOS_DIR / 'vif_regresion.csv', index=False)

# Coeficientes e interpretación
coef_reg = pd.DataFrame({
    'variable': VARS_REGRESION,
    'coeficiente': lreg_model.coef_
}).sort_values('coeficiente', key=abs, ascending=False)
print("\n  Coeficientes de regresión:")
print(coef_reg.to_string(index=False))
coef_reg.to_csv(RESULTADOS_DIR / 'coeficientes_regresion.csv', index=False)

# ── Gráficos de regresión lineal ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Regresión Lineal — Producción  (R²={r2:.3f})', fontsize=14, fontweight='bold')

# Real vs predicho
axes[0].scatter(y_reg_test, y_reg_pred, alpha=0.5, s=20, color='#3498db')
lim = [min(y_reg_test.min(), y_reg_pred.min()),
       max(y_reg_test.max(), y_reg_pred.max())]
axes[0].plot(lim, lim, 'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlabel('Producción Real (unidades)')
axes[0].set_ylabel('Producción Predicha (unidades)')
axes[0].set_title('Real vs Predicho')
axes[0].legend()

# Residuos
residuos = y_reg_test - y_reg_pred
axes[1].scatter(y_reg_pred, residuos, alpha=0.5, s=20, color='#9b59b6')
axes[1].axhline(0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel('Valores Predichos')
axes[1].set_ylabel('Residuos')
axes[1].set_title('Residuos vs Predichos')

# Coeficientes
coef_plot = coef_reg.sort_values('coeficiente')
colores_coef_reg = ['#e74c3c' if v > 0 else '#27ae60' for v in coef_plot['coeficiente']]
axes[2].barh(coef_plot['variable'], coef_plot['coeficiente'], color=colores_coef_reg)
axes[2].axvline(0, color='black', lw=0.8)
axes[2].set_title('Coeficientes del Modelo')
axes[2].set_xlabel('Coeficiente')

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'modelo_04_regresion_lineal.png', bbox_inches='tight')
plt.close()
print("  ✓ modelo_04_regresion_lineal.png guardado")

joblib.dump(lreg_model, RESULTADOS_DIR / 'modelo_linear_regression.pkl')
print("  ✓ Modelo guardado: modelo_linear_regression.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: ANÁLISIS DE CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n[10/13] Análisis de correlación con variable objetivo...")

cols_analisis = VARS_NUMERICAS + ['falla_maquina']
corr_target = df[cols_analisis].corr()['falla_maquina'].drop('falla_maquina').sort_values()

print("\n  Correlación de Pearson con falla_maquina:")
print(corr_target.round(4).to_string())

# ── Gráfico de correlaciones con el target ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Análisis de Correlación con Falla de Maquinaria', fontsize=13, fontweight='bold')

colores_corr = ['#e74c3c' if v > 0 else '#27ae60' for v in corr_target]
corr_target.plot(kind='barh', ax=axes[0], color=colores_corr, edgecolor='white')
axes[0].axvline(0, color='black', lw=0.8)
axes[0].set_title('Correlación con falla_maquina')
axes[0].set_xlabel('Coeficiente de Pearson')
for i, v in enumerate(corr_target):
    axes[0].text(v + (0.005 if v > 0 else -0.005), i,
                 f'{v:.3f}', va='center', ha='left' if v > 0 else 'right', fontsize=8)

# Pairplot simplificado
vars_par = ['temperatura_c', 'vibracion_mm_s', 'dias_desde_mantenimiento', 'falla_maquina']
scatter_data = df[vars_par].copy()
c_map = {0: '#27ae60', 1: '#e74c3c'}
colores_scatter = scatter_data['falla_maquina'].map(c_map)
axes[1].scatter(scatter_data['temperatura_c'],
                scatter_data['dias_desde_mantenimiento'],
                c=colores_scatter, alpha=0.5, s=20)
axes[1].set_xlabel('Temperatura (°C)')
axes[1].set_ylabel('Días desde Mantenimiento')
axes[1].set_title('Temperatura vs Mantenimiento\n(verde=OK, rojo=Falla)')

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'analisis_correlacion.png', bbox_inches='tight')
plt.close()
print("  ✓ analisis_correlacion.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: COMPARACIÓN DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[11/13] Comparando modelos de clasificación...")

metricas_df = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Árbol de Decisión'],
    'Accuracy': [acc_lr, acc_dt],
    'Precision': [prec_lr, prec_dt],
    'Recall': [rec_lr, rec_dt],
    'F1-Score': [f1_lr, f1_dt],
    'AUC-ROC': [roc_auc_lr, roc_auc_dt],
    'CV-5 Accuracy': [cv_scores_lr.mean(), cv_scores_dt.mean()],
})

print("\n  Tabla comparativa:")
print(metricas_df.round(4).to_string(index=False))
metricas_df.to_csv(RESULTADOS_DIR / 'metricas_modelos.csv', index=False)
print(f"\n  ✓ Tabla de métricas guardada")

# ── Gráfico comparativo ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
metricas_comparar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(metricas_comparar))
width = 0.35

bars1 = ax.bar(x - width/2, metricas_df.loc[0, metricas_comparar],
               width, label='Reg. Logística', color='#2980b9', edgecolor='white')
bars2 = ax.bar(x + width/2, metricas_df.loc[1, metricas_comparar],
               width, label='Árbol de Decisión', color='#8e44ad', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(metricas_comparar)
ax.set_ylabel('Valor (0 - 1)')
ax.set_title('Comparación de Métricas — Modelos de Clasificación', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend()
ax.axhline(0.75, color='gray', linestyle=':', lw=1, label='Umbral 75%')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / 'comparacion_modelos.png', bbox_inches='tight')
plt.close()
print("  ✓ comparacion_modelos.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13: EXPORTACIÓN DE RESULTADOS FINALES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[12/13] Exportando resultados finales...")

# Predicciones de probabilidad sobre todo el dataset
X_all = df_clean[VARS_MODELO]
X_all_sc = scaler.transform(X_all)
proba_todas = lr_model.predict_proba(X_all_sc)[:, 1]
df_clean['probabilidad_falla'] = proba_todas
df_clean['prediccion_falla']   = lr_model.predict(X_all_sc)

# Máquinas de alto riesgo (prob > 0.70)
cols_salida = ['id_maquina', 'tipo_maquina', 'planta', 'turno',
               'probabilidad_falla', 'cluster', 'cluster_nombre',
               'temperatura_c', 'vibracion_mm_s', 'presion_bar',
               'dias_desde_mantenimiento', 'antiguedad_anios',
               'tasa_defectos_pct', 'horas_operacion', 'falla_maquina']

# Filtrar con columnas que existen
cols_disponibles = [c for c in cols_salida if c in df_clean.columns]
alto_riesgo = (
    df_clean[df_clean['probabilidad_falla'] > 0.70]
    [cols_disponibles]
    .sort_values('probabilidad_falla', ascending=False)
)

alto_riesgo.to_csv(RESULTADOS_DIR / 'maquinas_alto_riesgo.csv', index=False)
print(f"  ✓ {len(alto_riesgo)} máquinas de alto riesgo (prob > 70%) identificadas")

# Dataset limpio con todas las predicciones
df_clean.to_csv(RESULTADOS_DIR / 'dataset_limpio.csv', index=False)

# Resumen ejecutivo
resumen = {
    'total_registros': [len(df)],
    'tasa_falla_pct': [round(tasa_falla, 2)],
    'accuracy_logistica': [round(acc_lr, 4)],
    'precision_logistica': [round(prec_lr, 4)],
    'recall_logistica': [round(rec_lr, 4)],
    'f1_logistica': [round(f1_lr, 4)],
    'auc_roc_logistica': [round(roc_auc_lr, 4)],
    'accuracy_arbol': [round(acc_dt, 4)],
    'f1_arbol': [round(f1_dt, 4)],
    'auc_roc_arbol': [round(roc_auc_dt, 4)],
    'r2_regresion_lineal': [round(r2, 4)],
    'rmse_regresion_lineal': [round(rmse, 2)],
    'silhouette_kmeans': [round(silhouette_final, 4)],
    'clusters_identificados': [k_final],
    'maquinas_alto_riesgo_n': [len(alto_riesgo)],
    'maquinas_alto_riesgo_pct': [round(len(alto_riesgo)/len(df)*100, 1)],
}
pd.DataFrame(resumen).to_csv(RESULTADOS_DIR / 'resumen_ejecutivo.csv', index=False)
print("  ✓ resumen_ejecutivo.csv guardado")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13 (continuación): DASHBOARD SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════

print("\n[13/13] Imprimiendo resumen de resultados...")
print("\n" + SEPARADOR)
print("  RESUMEN DE RESULTADOS — PROYECTO MANUFACTURA")
print(SEPARADOR)
print(f"\n  {'Dataset':}")
print(f"    Registros analizados      : {len(df):,}")
print(f"    Variables                 : {df.shape[1]}")
print(f"    Tasa de falla             : {tasa_falla:.1f}%")
print(f"\n  {'Modelo 1: Regresión Logística':}")
print(f"    Accuracy                  : {acc_lr:.1%}")
print(f"    Recall (sensibilidad)     : {rec_lr:.1%}  → detecta {rec_lr*100:.0f} de cada 100 fallas")
print(f"    AUC-ROC                   : {roc_auc_lr:.3f}")
print(f"\n  {'Modelo 2: K-Means Clustering':}")
print(f"    Clusters identificados    : {k_final}")
print(f"    Silhouette score          : {silhouette_final:.4f}")
print(f"    Segmentos                 : {list(nombres_cluster.values())}")
print(f"\n  {'Modelo 3: Árbol de Decisión':}")
print(f"    Accuracy                  : {acc_dt:.1%}")
print(f"    AUC-ROC                   : {roc_auc_dt:.3f}")
top_feat = importancias.iloc[0]
print(f"    Variable más importante   : {top_feat['variable']} ({top_feat['importancia']:.3f})")
print(f"\n  {'Modelo 4: Regresión Lineal':}")
print(f"    R² Score                  : {r2:.4f}")
print(f"    RMSE                      : {rmse:.1f} unidades")
top_coef = coef_reg.iloc[0]
print(f"    Mayor impacto en producción: {top_coef['variable']} ({top_coef['coeficiente']:+.2f})")
print(f"\n  {'Resultados de Negocio':}")
print(f"    Máquinas de alto riesgo   : {len(alto_riesgo)} ({len(alto_riesgo)/len(df)*100:.1f}%)")
print(f"    Archivos generados        : {len(list(RESULTADOS_DIR.glob('*')))} archivos en resultados/manufactura/")
print(f"\n  Resultados guardados en: {RESULTADOS_DIR}")
print("\n" + SEPARADOR)
print("  ✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
print(SEPARADOR + "\n")
