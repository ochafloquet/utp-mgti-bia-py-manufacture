"""
Dashboard Manufactura — UTP MGTI BIA
Proyecto Final: Análisis de Operaciones Industriales

Ejecutar:
    streamlit run scripts/dashboard_manufactura.py
"""

import sys
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Manufactura — UTP BIA",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Rutas
BASE_DIR = Path(__file__).parent.parent
RES_DIR  = BASE_DIR / 'resultados' / 'manufactura'

# ── Carga de datos ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df         = pd.read_csv(RES_DIR / 'dataset_limpio.csv')
    alto_riesgo = pd.read_csv(RES_DIR / 'maquinas_alto_riesgo.csv')
    metricas   = pd.read_csv(RES_DIR / 'metricas_modelos.csv')
    segmentos  = pd.read_csv(RES_DIR / 'segmentos_clustering.csv')
    resumen    = pd.read_csv(RES_DIR / 'resumen_ejecutivo.csv')
    return df, alto_riesgo, metricas, segmentos, resumen

try:
    df, alto_riesgo, metricas, segmentos, resumen = load_data()
    DATA_OK = True
except FileNotFoundError:
    DATA_OK = False

# ── CSS personalizado ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ENCABEZADO
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏭 Dashboard de Manufactura Industrial")
st.markdown("""
**UTP — Maestría en Gerencia de TI y Transformación Digital**  
*Business Intelligence & Analytics — Proyecto Final*
""")
st.divider()

if not DATA_OK:
    st.error("⚠️ No se encontraron los datos. Ejecuta primero: `python scripts/analisis_manufactura.py`")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTROS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔧 Filtros")

    plantas_disp = ['Todas'] + sorted(df['planta'].dropna().unique().tolist())
    planta_sel = st.selectbox("Planta", plantas_disp)

    tipos_disp = ['Todos'] + sorted(df['tipo_maquina'].dropna().unique().tolist())
    tipo_sel = st.selectbox("Tipo de Máquina", tipos_disp)

    turnos_disp = ['Todos'] + sorted(df['turno'].dropna().unique().tolist())
    turno_sel = st.selectbox("Turno", turnos_disp)

    st.divider()
    st.info("💡 Los filtros aplican a todas las páginas excepto la de Modelos.")

# Aplicar filtros
df_fil = df.copy()
if planta_sel != 'Todas':
    df_fil = df_fil[df_fil['planta'] == planta_sel]
if tipo_sel != 'Todos':
    df_fil = df_fil[df_fil['tipo_maquina'] == tipo_sel]
if turno_sel != 'Todos':
    df_fil = df_fil[df_fil['turno'] == turno_sel]

# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vista General",
    "⚠️ Predicción de Fallas",
    "🔵 Segmentación",
    "📈 Eficiencia Operativa",
    "🤖 Modelos",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: VISTA GENERAL
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("KPIs Principales")

    tasa_falla   = df_fil['falla_maquina'].mean() * 100
    n_maquinas   = df_fil['id_maquina'].nunique() if 'id_maquina' in df_fil.columns else len(df_fil)
    prod_media   = df_fil['produccion_unidades'].mean()
    defectos_med = df_fil['tasa_defectos_pct'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📋 Registros",   f"{len(df_fil):,}")
    col2.metric("🔧 Máquinas",    f"{n_maquinas:,}")
    col3.metric("⚠️ Tasa de Falla", f"{tasa_falla:.1f}%",
                delta=f"{tasa_falla - df['falla_maquina'].mean()*100:.1f}% vs total",
                delta_color="inverse")
    col4.metric("📦 Prod. Media", f"{prod_media:.0f} u/turno")
    col5.metric("❌ Defectos Med", f"{defectos_med:.1f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.pie(
            df_fil,
            names=df_fil['falla_maquina'].map({0: 'Sin Falla', 1: 'Con Falla'}),
            title="Distribución de Fallas",
            color_discrete_map={'Sin Falla': '#27ae60', 'Con Falla': '#e74c3c'},
            hole=0.4,
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fallas_planta = (
            df_fil.groupby('planta')['falla_maquina']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'tasa_falla', 'count': 'total'})
            .reset_index()
        )
        fallas_planta['tasa_falla_pct'] = fallas_planta['tasa_falla'] * 100
        fig2 = px.bar(
            fallas_planta, x='planta', y='tasa_falla_pct',
            color='tasa_falla_pct',
            color_continuous_scale='RdYlGn_r',
            title="Tasa de Falla por Planta (%)",
            labels={'tasa_falla_pct': 'Tasa de Falla (%)', 'planta': 'Planta'},
            text_auto='.1f',
        )
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fallas_tipo = (
            df_fil.groupby('tipo_maquina')['falla_maquina']
            .mean().reset_index()
            .rename(columns={'falla_maquina': 'tasa_falla'})
        )
        fallas_tipo['tasa_falla_pct'] = fallas_tipo['tasa_falla'] * 100
        fig3 = px.bar(
            fallas_tipo.sort_values('tasa_falla_pct', ascending=True),
            x='tasa_falla_pct', y='tipo_maquina', orientation='h',
            color='tasa_falla_pct', color_continuous_scale='RdYlGn_r',
            title="Tasa de Falla por Tipo de Máquina",
            labels={'tasa_falla_pct': 'Tasa (%)', 'tipo_maquina': 'Tipo'},
            text_auto='.1f',
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fallas_turno = (
            df_fil.groupby('turno')['falla_maquina']
            .mean().reset_index()
            .rename(columns={'falla_maquina': 'tasa_falla'})
        )
        fallas_turno['tasa_falla_pct'] = fallas_turno['tasa_falla'] * 100
        fig4 = px.bar(
            fallas_turno.sort_values('tasa_falla_pct', ascending=False),
            x='turno', y='tasa_falla_pct',
            color='tasa_falla_pct', color_continuous_scale='RdYlGn_r',
            title="Tasa de Falla por Turno",
            labels={'tasa_falla_pct': 'Tasa (%)', 'turno': 'Turno'},
            text_auto='.1f',
        )
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PREDICCIÓN DE FALLAS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("⚠️ Predicción de Fallas — Máquinas de Alto Riesgo")

    ar_fil = alto_riesgo.copy()
    if planta_sel != 'Todas' and 'planta' in ar_fil.columns:
        ar_fil = ar_fil[ar_fil['planta'] == planta_sel]
    if tipo_sel != 'Todos' and 'tipo_maquina' in ar_fil.columns:
        ar_fil = ar_fil[ar_fil['tipo_maquina'] == tipo_sel]

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Alto Riesgo (>70%)", f"{len(ar_fil):,}")
    col2.metric("📊 Prob. Media de Falla", f"{ar_fil['probabilidad_falla'].mean()*100:.1f}%"
                if len(ar_fil) > 0 else "N/A")
    col3.metric("🔝 Prob. Máxima", f"{ar_fil['probabilidad_falla'].max()*100:.1f}%"
                if len(ar_fil) > 0 else "N/A")

    if len(ar_fil) > 0:
        fig5 = px.histogram(
            ar_fil, x='probabilidad_falla',
            nbins=20,
            color_discrete_sequence=['#e74c3c'],
            title="Distribución de Probabilidades de Falla (>70%)",
            labels={'probabilidad_falla': 'Probabilidad de Falla', 'count': 'N° Máquinas'},
        )
        fig5.add_vline(x=0.8, line_dash='dash', line_color='orange', annotation_text='Riesgo crítico 80%')
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader(f"Top 50 Máquinas de Mayor Riesgo ({len(ar_fil)} total)")
        display_cols = [c for c in ['id_maquina', 'tipo_maquina', 'planta', 'turno',
                                     'probabilidad_falla', 'temperatura_c', 'vibracion_mm_s',
                                     'dias_desde_mantenimiento', 'cluster_nombre']
                        if c in ar_fil.columns]

        ar_show = ar_fil[display_cols].head(50).copy()
        if 'probabilidad_falla' in ar_show.columns:
            ar_show['probabilidad_falla'] = ar_show['probabilidad_falla'].map('{:.1%}'.format)

        st.dataframe(ar_show, use_container_width=True, hide_index=True)

        csv_bytes = ar_fil.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Descargar lista completa (CSV)",
            data=csv_bytes,
            file_name='maquinas_alto_riesgo.csv',
            mime='text/csv',
        )
    else:
        st.info("No hay máquinas de alto riesgo con los filtros seleccionados.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: SEGMENTACIÓN K-MEANS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔵 Segmentación Operativa — K-Means (k=4)")

    cluster_nombres = {0: 'Eficientes', 1: 'Alerta', 2: 'Estándar', 3: 'Críticas'}
    cluster_colores = {'Eficientes': '#27ae60', 'Estándar': '#3498db',
                       'Alerta': '#f39c12', 'Críticas': '#e74c3c'}

    if 'cluster_nombre' in df_fil.columns:
        dist_cluster = df_fil['cluster_nombre'].value_counts().reset_index()
        dist_cluster.columns = ['Segmento', 'Cantidad']

        col1, col2 = st.columns(2)
        with col1:
            fig6 = px.pie(
                dist_cluster, names='Segmento', values='Cantidad',
                color='Segmento',
                color_discrete_map=cluster_colores,
                title="Distribución de Máquinas por Segmento",
                hole=0.35,
            )
            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            seg_stats = df_fil.groupby('cluster_nombre').agg(
                N=('falla_maquina', 'count'),
                tasa_falla=('falla_maquina', 'mean'),
                temp_media=('temperatura_c', 'mean'),
                vibracion_media=('vibracion_mm_s', 'mean'),
                dias_mant=('dias_desde_mantenimiento', 'mean'),
                prod_media=('produccion_unidades', 'mean'),
            ).round(2).reset_index()
            seg_stats['tasa_falla'] = (seg_stats['tasa_falla'] * 100).round(1)

            fig7 = px.bar(
                seg_stats, x='cluster_nombre', y='tasa_falla',
                color='cluster_nombre', color_discrete_map=cluster_colores,
                title="Tasa de Falla por Segmento (%)",
                labels={'cluster_nombre': 'Segmento', 'tasa_falla': 'Tasa (%)'},
                text_auto='.1f',
            )
            fig7.update_layout(showlegend=False)
            st.plotly_chart(fig7, use_container_width=True)

        st.subheader("Perfil de Segmentos")
        st.dataframe(seg_stats.rename(columns={
            'cluster_nombre': 'Segmento', 'N': 'N° Máquinas',
            'tasa_falla': 'Tasa Falla (%)', 'temp_media': 'Temp. Media (°C)',
            'vibracion_media': 'Vibración Media', 'dias_mant': 'Días sin Mant.',
            'prod_media': 'Prod. Media'
        }), use_container_width=True, hide_index=True)

        # Scatter con clusters
        fig8 = px.scatter(
            df_fil, x='temperatura_c', y='vibracion_mm_s',
            color='cluster_nombre',
            color_discrete_map=cluster_colores,
            title="Temperatura vs Vibración — Segmentos Operativos",
            labels={'temperatura_c': 'Temperatura (°C)', 'vibracion_mm_s': 'Vibración (mm/s)',
                    'cluster_nombre': 'Segmento'},
            hover_data=['dias_desde_mantenimiento', 'tasa_defectos_pct'],
            opacity=0.7,
        )
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Ejecuta el análisis para ver la segmentación.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: EFICIENCIA OPERATIVA
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 Eficiencia Operativa")

    col1, col2 = st.columns(2)
    with col1:
        prod_tipo = df_fil.groupby('tipo_maquina')['produccion_unidades'].mean().reset_index()
        fig9 = px.bar(
            prod_tipo.sort_values('produccion_unidades', ascending=True),
            x='produccion_unidades', y='tipo_maquina', orientation='h',
            color='produccion_unidades', color_continuous_scale='Greens',
            title="Producción Media por Tipo (unidades)",
            text_auto='.0f',
        )
        fig9.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig9, use_container_width=True)

    with col2:
        energia_planta = df_fil.groupby('planta')['consumo_energia_kwh'].mean().reset_index()
        fig10 = px.bar(
            energia_planta.sort_values('consumo_energia_kwh', ascending=False),
            x='planta', y='consumo_energia_kwh',
            color='consumo_energia_kwh', color_continuous_scale='Oranges',
            title="Consumo Energético Medio por Planta (kWh)",
            text_auto='.0f',
        )
        fig10.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig10, use_container_width=True)

    # Scatter producción vs consumo energético
    fig11 = px.scatter(
        df_fil, x='consumo_energia_kwh', y='produccion_unidades',
        color='tipo_maquina',
        title="Consumo Energético vs Producción",
        labels={'consumo_energia_kwh': 'Energía (kWh)', 'produccion_unidades': 'Producción (u)'},
        trendline='ols',
        opacity=0.6,
    )
    st.plotly_chart(fig11, use_container_width=True)

    # Defectos
    fig12 = px.box(
        df_fil, x='tipo_maquina', y='tasa_defectos_pct',
        color='tipo_maquina',
        title="Distribución de Tasa de Defectos por Tipo de Máquina",
        labels={'tasa_defectos_pct': 'Tasa de Defectos (%)', 'tipo_maquina': 'Tipo'},
    )
    fig12.update_layout(showlegend=False)
    st.plotly_chart(fig12, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: MODELOS — MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🤖 Métricas de Modelos de Machine Learning")

    # KPIs de modelos
    res = resumen.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Logística — Accuracy",  f"{res['accuracy_logistica']*100:.1f}%")
    col1.metric("Logística — AUC-ROC",   f"{res['auc_roc_logistica']:.3f}")
    col2.metric("Logística — Recall",    f"{res['recall_logistica']*100:.1f}%")
    col2.metric("Logística — F1",        f"{res['f1_logistica']:.3f}")
    col3.metric("Árbol — Accuracy",      f"{res['accuracy_arbol']*100:.1f}%")
    col3.metric("Árbol — AUC-ROC",       f"{res['auc_roc_arbol']:.3f}")
    col4.metric("Reg. Lineal — R²",      f"{res['r2_regresion_lineal']:.4f}")
    col4.metric("K-Means — Silhouette",  f"{res['silhouette_kmeans']:.4f}")

    st.divider()

    # Tabla de métricas
    st.subheader("Comparación de Modelos de Clasificación")
    st.dataframe(metricas, use_container_width=True, hide_index=True)

    # Gráfico comparativo
    metricas_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metricas_cols_disp = [c for c in metricas_cols if c in metricas.columns]

    if metricas_cols_disp:
        fig_comp = go.Figure()
        for _, row in metricas.iterrows():
            fig_comp.add_trace(go.Bar(
                name=row['Modelo'],
                x=metricas_cols_disp,
                y=[row[c] for c in metricas_cols_disp],
                text=[f"{row[c]:.3f}" for c in metricas_cols_disp],
                textposition='outside',
            ))
        fig_comp.update_layout(
            barmode='group',
            title='Comparación de Métricas por Modelo',
            yaxis_range=[0, 1.15],
            yaxis_title='Valor (0-1)',
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()
    st.subheader("Interpretación de Resultados")
    st.markdown(f"""
    | Modelo | Métrica Clave | Valor | Interpretación |
    |--------|--------------|-------|----------------|
    | Regresión Logística | AUC-ROC | {res['auc_roc_logistica']:.3f} | Buena discriminación entre falla/no-falla |
    | Regresión Logística | Recall | {res['recall_logistica']*100:.1f}% | Detecta este % de fallas reales |
    | Árbol de Decisión | Accuracy | {res['accuracy_arbol']*100:.1f}% | Más interpretable, reglas explícitas |
    | K-Means | Silhouette | {res['silhouette_kmeans']:.4f} | Separación moderada entre segmentos |
    | Regresión Lineal | R² | {res['r2_regresion_lineal']:.4f} | Producción no depende linealmente de estas vars. |

    **Recomendación:** Usar **Regresión Logística** para priorizar mantenimiento preventivo 
    y el **Árbol de Decisión** para comunicar reglas claras al equipo operativo.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.8rem;'>
UTP — Maestría en Gerencia de TI y Transformación Digital | Business Intelligence & Analytics | 2025
</div>
""", unsafe_allow_html=True)
