"""
Generador de Dataset Sintetico - Manufactura Industrial
UTP MGTI BIA - Proyecto Final

Ejecutar si no existe dataset_manufactura_1000.csv:
    python data/manufactura/generar_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 1000
PLANTAS = ['Lima', 'Arequipa', 'Trujillo']
TIPOS = ['Prensa', 'Torno CNC', 'Inyectora', 'Empacadora', 'Compresor']
TURNOS = ['Mañana', 'Tarde', 'Noche']

def generar_dataset():
    ids_maquina = [f"MQ-{str(i).zfill(3)}" for i in range(1, 181)]
    data = {
        'id_registro':          [f"MAN-{str(i).zfill(4)}" for i in range(1, N+1)],
        'id_maquina':           np.random.choice(ids_maquina, N),
        'planta':               np.random.choice(PLANTAS, N, p=[0.5, 0.3, 0.2]),
        'tipo_maquina':         np.random.choice(TIPOS, N),
        'turno':                np.random.choice(TURNOS, N),
        'temperatura_c':        np.round(np.random.uniform(45, 109, N), 1),
        'vibracion_mm_s':       np.round(np.random.uniform(0.5, 9.4, N), 2),
        'presion_bar':          np.round(np.random.uniform(2, 12, N), 1),
        'horas_operacion':      np.round(np.random.uniform(100, 12000, N)).astype(int),
        'antiguedad_anios':     np.round(np.random.uniform(0.2, 18, N), 1),
        'dias_desde_mantenimiento': np.round(np.random.uniform(1, 240, N)).astype(int),
        'tasa_defectos_pct':    np.round(np.random.uniform(0.1, 18, N), 1),
        'consumo_energia_kwh':  np.round(np.random.uniform(247, 950, N), 2),
        'produccion_unidades':  np.round(np.random.uniform(150, 1400, N)).astype(int),
    }
    df = pd.DataFrame(data)
    prob_falla = (
        0.3 * (df['temperatura_c'] - 45) / (109 - 45) +
        0.3 * (df['vibracion_mm_s'] - 0.5) / (9.4 - 0.5) +
        0.2 * (df['dias_desde_mantenimiento'] - 1) / (240 - 1) +
        0.1 * (df['tasa_defectos_pct'] - 0.1) / (18 - 0.1) +
        0.1 * (df['antiguedad_anios'] - 0.2) / (18 - 0.2)
    ).clip(0, 1)
    df['falla_maquina'] = (np.random.random(N) < prob_falla).astype(int)
    df['produccion_unidades'] = (
        df['produccion_unidades'] *
        (1 - 0.3 * df['falla_maquina']) *
        (1 - 0.015 * df['tasa_defectos_pct'])
    ).round().astype(int).clip(150, 1400)
    out_path = Path(__file__).parent / 'dataset_manufactura_1000.csv'
    df.to_csv(out_path, index=False)
    print(f"Dataset generado: {out_path}  |  Shape: {df.shape}  |  Tasa falla: {df['falla_maquina'].mean()*100:.1f}%")
    return df

if __name__ == '__main__':
    generar_dataset()
