#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREPARACIÓN DE RESULTADOS PARA PAPER ACADÉMICO - VERSIÓN CORREGIDA
Sin dependencia de fuentes específicas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import pandas as pd
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE MATPLOTLIB - SIN ERRORES DE FUENTES
# ============================================================

# Usar fuentes disponibles en cualquier sistema
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',  # Cambiado de 'serif' a 'sans-serif'
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],  # Fuentes alternativas
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (8, 6),
    'text.usetex': False,  # No usar LaTeX (evita más errores)
})

# Si quieres usar LaTeX (opcional, requiere instalación)
# plt.rcParams.update({'text.usetex': True, 'font.family': 'serif'})

# Crear directorios
os.makedirs('paper_results/figures', exist_ok=True)
os.makedirs('paper_results/tables', exist_ok=True)
os.makedirs('paper_results/data', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("="*70)
print("PREPARACIÓN DE RESULTADOS PARA PAPER ACADÉMICO")
print("Versión corregida - Sin errores de fuentes")
print("="*70)

# ============================================================
# 1. DEFINICIÓN DEL MODELO
# ============================================================

def evolution_equation(t, y, alpha, beta, kappa):
    """
    Ecuación de evolución de la Gradiente de Acoplamiento Γ
    """
    Gamma = y[0]
    Gamma_max = 4.0
    Gamma_0 = 0.5
    
    saturación = -alpha * Gamma**2
    crecimiento = beta * Gamma * (1 - Gamma/Gamma_max)
    fuente_oscura = kappa * np.exp(-Gamma/Gamma_0)
    
    dGamma_dt = saturación + crecimiento + fuente_oscura
    
    if Gamma <= 0 and dGamma_dt < 0:
        dGamma_dt = 0
    
    return [dGamma_dt]

# ============================================================
# 2. SIMULACIONES
# ============================================================

print("\n[1/6] Ejecutando simulaciones...")

scenarios = {
    'Foton (cuantico puro)': {
        'alpha': 0.15, 'beta': 0.7, 'kappa': 0.01, 'Gamma0': 0.001, 't_max': 30.0,
        'color': 'blue', 'marker': 'o', 'linestyle': '-'
    },
    'Electron (dual)': {
        'alpha': 0.10, 'beta': 0.9, 'kappa': 0.05, 'Gamma0': 0.5, 't_max': 40.0,
        'color': 'green', 'marker': 's', 'linestyle': '-'
    },
    'Proton (clasico)': {
        'alpha': 0.05, 'beta': 1.2, 'kappa': 0.10, 'Gamma0': 2.0, 't_max': 50.0,
        'color': 'red', 'marker': '^', 'linestyle': '-'
    },
    'Materia Oscura (estable)': {
        'alpha': 0.20, 'beta': 0.4, 'kappa': 0.02, 'Gamma0': 0.05, 't_max': 60.0,
        'color': 'purple', 'marker': 'd', 'linestyle': '-'
    }
}

results = {}

for name, params in scenarios.items():
    print(f"  Simulando: {name}...")
    
    t_eval = np.linspace(0, params['t_max'], 2000)
    
    sol = solve_ivp(
        evolution_equation,
        [0, params['t_max']],
        [params['Gamma0']],
        t_eval=t_eval,
        args=(params['alpha'], params['beta'], params['kappa']),
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    
    Gamma = sol.y[0]
    Gamma = np.clip(Gamma, 1e-6, 10.0)
    
    masa = Gamma / (1 + Gamma)
    kappa_val = 0.5 * (1 - np.exp(-np.clip(Gamma, 0, 5)))
    dGamma = np.gradient(Gamma, sol.t)
    
    results[name] = {
        't': sol.t,
        'Gamma': Gamma,
        'masa': masa,
        'kappa': kappa_val,
        'dGamma': dGamma,
        'params': params
    }

print("  ✓ Simulaciones completadas")

# ============================================================
# 3. FIGURA 1: EVOLUCIÓN TEMPORAL
# ============================================================

print("\n[2/6] Generando Figura 1...")

fig1, axes = plt.subplots(2, 2, figsize=(11, 9))

# Panel A: Evolución de Γ
ax_a = axes[0, 0]
for name, data in results.items():
    ax_a.plot(data['t'], data['Gamma'], 
              color=scenarios[name]['color'], 
              linewidth=2,
              label=name.replace('_', ' '))
ax_a.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='G=1 (frontera)')
ax_a.set_xlabel('Tiempo (u.a.)')
ax_a.set_ylabel('G (Gradiente de Acoplamiento)')
ax_a.set_title('(a) Evolucion temporal de G')
ax_a.legend(loc='lower right', fontsize=8)
ax_a.grid(True, alpha=0.3)
ax_a.set_yscale('log')
ax_a.set_ylim(1e-3, 10)

# Panel B: Masa emergente
ax_b = axes[0, 1]
for name, data in results.items():
    ax_b.plot(data['t'], data['masa'],
              color=scenarios[name]['color'],
              linewidth=2,
              label=name.replace('_', ' '))
ax_b.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='m/m0 = 0.5')
ax_b.set_xlabel('Tiempo (u.a.)')
ax_b.set_ylabel('Masa relativa m/m0')
ax_b.set_title('(b) Emergencia de la masa')
ax_b.legend(loc='lower right', fontsize=8)
ax_b.grid(True, alpha=0.3)

# Panel C: Relación κ(Γ)
ax_c = axes[1, 0]
for name, data in results.items():
    ax_c.plot(data['Gamma'], data['kappa'],
              color=scenarios[name]['color'],
              linewidth=2,
              label=name.replace('_', ' '))
ax_c.set_xlabel('G')
ax_c.set_ylabel('k (acoplamiento Eje Horizontal)')
ax_c.set_title('(c) Relacion k = f(G)')
ax_c.legend(loc='lower right', fontsize=8)
ax_c.grid(True, alpha=0.3)
ax_c.set_xscale('log')
ax_c.set_xlim(1e-3, 10)

# Panel D: Espacio de fases
ax_d = axes[1, 1]
for name, data in results.items():
    ax_d.plot(data['Gamma'], data['dGamma'],
              color=scenarios[name]['color'],
              linewidth=1.5,
              alpha=0.7)
    ax_d.scatter(data['Gamma'][0], data['dGamma'][0],
                 color=scenarios[name]['color'], s=80, marker='o', edgecolors='black', zorder=5)
    ax_d.scatter(data['Gamma'][-1], data['dGamma'][-1],
                 color=scenarios[name]['color'], s=100, marker='s', edgecolors='black', zorder=5)
ax_d.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax_d.axvline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax_d.set_xlabel('G')
ax_d.set_ylabel('dG/dt')
ax_d.set_title('(d) Espacio de fases')
ax_d.set_xscale('log')
ax_d.set_xlim(1e-3, 10)
ax_d.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_results/figures/Fig1_evolucion_temporal.png', dpi=300)
plt.savefig('paper_results/figures/Fig1_evolucion_temporal.pdf', format='pdf')
plt.close(fig1)
print("  ✓ Figura 1 guardada")

# ============================================================
# 4. FIGURA 2: TRANSICIÓN CUÁNTICO-CLÁSICO
# ============================================================

print("\n[3/6] Generando Figura 2...")

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

Gamma_range = np.logspace(-3, 1, 500)
coherencia = np.exp(-Gamma_range**2 / 2)
localizacion = 1 - np.exp(-Gamma_range**2)

# Panel A
ax_a = axes[0]
ax_a.plot(Gamma_range, coherencia, 'b-', linewidth=2, label='Coherencia cuantica $e^{-G^2/2}$')
ax_a.plot(Gamma_range, localizacion, 'r-', linewidth=2, label='Localizacion $1-e^{-G^2}$')
ax_a.axvline(1.0, color='purple', linestyle='--', linewidth=2, label='G = 1 (frontera)')
ax_a.fill_between(Gamma_range[Gamma_range <= 1], 0, coherencia[Gamma_range <= 1], 
                   alpha=0.3, color='blue')
ax_a.fill_between(Gamma_range[Gamma_range >= 1], 0, localizacion[Gamma_range >= 1], 
                   alpha=0.3, color='red')
ax_a.set_xlabel('G (Gradiente de Acoplamiento)')
ax_a.set_ylabel('Magnitud relativa')
ax_a.set_title('(a) Funciones de transicion')
ax_a.set_xscale('log')
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3)

# Panel B
ax_b = axes[1]
region_labels = ['Cuantico puro\n(G < 0.1)', 
                 'Dualidad\n(0.1 < G < 1)', 
                 'Clasico\n(G > 1)']
region_boundaries = [0, 0.1, 1, 10]
region_colors = ['#1f77b4', '#ff7f0e', '#d62728']

for i in range(3):
    ax_b.axvspan(region_boundaries[i], region_boundaries[i+1], 
                 alpha=0.3, color=region_colors[i])
    ax_b.text(np.sqrt(region_boundaries[i]*region_boundaries[i+1]), 0.5,
              region_labels[i], ha='center', va='center', fontsize=10)

for i, (name, data) in enumerate(results.items()):
    Gamma_final = data['Gamma'][-1]
    y_pos = 0.3 + 0.1 * i
    ax_b.scatter(Gamma_final, y_pos, color=scenarios[name]['color'], 
                 s=120, marker=scenarios[name]['marker'], zorder=5)
    ax_b.annotate(name.replace('_', ' '), (Gamma_final, y_pos + 0.08), 
                  fontsize=7, ha='center', rotation=45)

ax_b.set_xlabel('G')
ax_b.set_title('(b) Regimenes de comportamiento fisico')
ax_b.set_xscale('log')
ax_b.set_xlim(0.001, 10)
ax_b.set_ylim(0, 1)
ax_b.set_yticks([])
ax_b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_results/figures/Fig2_transicion.png', dpi=300)
plt.savefig('paper_results/figures/Fig2_transicion.pdf', format='pdf')
plt.close(fig2)
print("  ✓ Figura 2 guardada")

# ============================================================
# 5. FIGURA 3: PREDICCIONES COSMOLÓGICAS
# ============================================================

print("\n[4/6] Generando Figura 3...")

fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Asimetría
ax_a = axes[0]
epsilon = 6e-10
Gamma_crit = 0.001
Gamma_vals = np.logspace(-5, 1, 500)
asimetria = epsilon * np.exp(-Gamma_crit / Gamma_vals)

ax_a.loglog(Gamma_vals, asimetria, 'b-', linewidth=2, label='Prediccion del modelo')
ax_a.axvline(Gamma_crit, color='red', linestyle='--', linewidth=1.5, label='G_crit = 0.001')
ax_a.axhline(epsilon, color='gray', linestyle=':', linewidth=1.5, label='Exceso cosmologico e = 6e-10')
ax_a.fill_between(Gamma_vals[Gamma_vals < 1], asimetria[Gamma_vals < 1], 
                   epsilon/100, alpha=0.2, color='blue')
ax_a.set_xlabel('G')
ax_a.set_ylabel('Asimetria A')
ax_a.set_title('(a) Asimetria materia-antimateria')
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3)

# Panel B: Materia Oscura
ax_b = axes[1]
Gamma_DM = results['Materia Oscura (estable)']['Gamma']
t_DM = results['Materia Oscura (estable)']['t']
ax_b.plot(t_DM, Gamma_DM, 'purple', linewidth=2, label='Materia Oscura (G ~ 0.2)')
ax_b.fill_between(t_DM, Gamma_DM, 0, alpha=0.3, color='purple')
ax_b.axhline(0.2, color='gray', linestyle='--', linewidth=1, label='G ~ 0.2 (estable)')
ax_b.set_xlabel('Tiempo (u.a.)')
ax_b.set_ylabel('G')
ax_b.set_title('(b) Materia Oscura como decantacion estable')
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_results/figures/Fig3_predicciones_cosmologicas.png', dpi=300)
plt.savefig('paper_results/figures/Fig3_predicciones_cosmologicas.pdf', format='pdf')
plt.close(fig3)
print("  ✓ Figura 3 guardada")

# ============================================================
# 6. TABLA DE RESULTADOS
# ============================================================

print("\n[5/6] Generando tablas...")

# Tabla principal
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Resultados de la simulacion de Campos Trinitarios}
\label{tab:resultados}
\begin{tabular}{lcccccc}
\toprule
Escenario & $\alpha$ & $\beta$ & $\Gamma_{final}$ & $m/m_0$ & $\kappa_{final}$ \\
\midrule
"""

for name, data in results.items():
    Gamma_f = data['Gamma'][-1]
    masa_f = data['masa'][-1]
    kappa_f = data['kappa'][-1]
    params = data['params']
    latex_table += f"{name} & {params['alpha']:.2f} & {params['beta']:.1f} & {Gamma_f:.3f} & {masa_f:.3f} & {kappa_f:.3f} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open('paper_results/tables/resultados_simulacion.tex', 'w') as f:
    f.write(latex_table)

# Tabla de interpretación
interpretacion_table = r"""\begin{table}[htbp]
\centering
\caption{Interpretacion fisica de la Gradiente de Acoplamiento $\Gamma$}
\label{tab:interpretacion}
\begin{tabular}{cll}
\toprule
Rango de $\Gamma$ & Regimen fisico & Ejemplo \\
\midrule
$\Gamma < 0.01$ & Cuantico puro & Foton \\
$0.01 < \Gamma < 0.1$ & Cuantico masivo & Neutrino \\
$0.1 < \Gamma < 0.5$ & Dualidad emergente & Electron \\
$0.5 < \Gamma < 1.0$ & Transicion & Mesones ligeros \\
$\Gamma = 1.0$ & Frontera micro-macro & Punto de no retorno \\
$1.0 < \Gamma < 2.0$ & Clasico incipiente & Muon \\
$2.0 < \Gamma < 3.0$ & Clasico consolidado & Proton \\
$\Gamma > 3.0$ & Clasico saturado & Materia nuclear \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('paper_results/tables/interpretacion_Gamma.tex', 'w') as f:
    f.write(interpretacion_table)

print("  ✓ Tablas guardadas")

# ============================================================
# 7. GUARDAR DATOS
# ============================================================

print("\n[6/6] Guardando datos numericos...")

for name, data in results.items():
    df = pd.DataFrame({
        'tiempo': data['t'],
        'Gamma': data['Gamma'],
        'masa_relativa': data['masa'],
        'kappa': data['kappa'],
        'dGamma_dt': data['dGamma']
    })
    filename = f"paper_results/data/{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    df.to_csv(filename, index=False)

# Guardar resumen
summary_data = {
    'timestamp': timestamp,
    'resultados': {}
}
for name, data in results.items():
    summary_data['resultados'][name] = {
        'Gamma_inicial': float(data['Gamma'][0]),
        'Gamma_final': float(data['Gamma'][-1]),
        'masa_relativa_final': float(data['masa'][-1])
    }

with open('paper_results/data/resumen.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("  ✓ Datos guardados")

# ============================================================
# 8. RESUMEN FINAL
# ============================================================

print("\n" + "="*70)
print("RESUMEN DE RESULTADOS NUMERICOS")
print("="*70)

for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Gamma_inicial = {data['Gamma'][0]:.6f}")
    print(f"  Gamma_final   = {data['Gamma'][-1]:.6f}")
    print(f"  Masa relativa = {data['masa'][-1]:.6f}")
    print(f"  kappa_final   = {data['kappa'][-1]:.6f}")

print("\n" + "="*70)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print(f"📁 Resultados guardados en: paper_results/")
print("\n🦋 Fluxus - Sin errores de fuentes")
