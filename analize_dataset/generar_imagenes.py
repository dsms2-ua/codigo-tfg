import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

alto = pd.read_excel("../dataset_processed/Dataset_Index.xlsx", sheet_name="Alto_Sax_Index")
tenor = pd.read_excel("../dataset_processed/Dataset_Index.xlsx", sheet_name="Tenor_Sax_Index")

ALTO_COLOR = "#2166AC"
TENOR_COLOR = "#D6640D"
GRIS = "#AAAAAA"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# Figura 1: Distribución por tipo de ejercicio
fig, ax = plt.subplots(figsize=(6, 4))
tipos = ['Melody', 'Exercise', 'Scale']

valores_alto = [alto['TYPE'].value_counts()[t] for t in tipos]
valores_tenor = [tenor['TYPE'].value_counts()[t] for t in tipos]

x = np.arange(len(tipos))
width = 0.35

ax.bar(x - width/2, valores_alto, width, label='Saxo Alto', color=ALTO_COLOR)
ax.bar(x + width/2, valores_tenor, width, label='Saxo Tenor', color=TENOR_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(tipos)
ax.set_ylabel('Número de Grabaciones')
ax.set_title('Distribución por Tipo de Pieza')
ax.legend()

for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    
plt.tight_layout()
plt.savefig("imagen1.pdf", bbox_inches='tight')
plt.savefig("imagen1.png", bbox_inches='tight')
plt.close()

# Figura 2: Distribución por compás
fig, axes = plt.subplots(1, 2, figsize=(6, 4))
for ax, df, label, color in [(axes[0], alto, 'Alto', ALTO_COLOR), (axes[1], tenor, 'Tenor', TENOR_COLOR)]:
    vc = df['MEASURE'].value_counts()
    top5 = vc.head(5)
    otros = vc.iloc[5:].sum()
    labels = list(top5.index) + ['Otros']
    sizes = list(top5.values) + [otros]
    colors_pie = [color] + [plt.cm.Blues(0.3 + 0.12*i) for i in range(4)] + [GRIS]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie[:len(labels)], pctdistance=0.78, textprops={'fontsize': 9})
    ax.set_title(f'Compás - {label}')

plt.tight_layout()
plt.savefig("imagen2.pdf", bbox_inches='tight')
plt.savefig("imagen2.png", bbox_inches='tight')
plt.close()

# Figura 3: Técnicas extendidas
fig, axes = plt.subplots(figsize=(8, 4))
tecnicas = ['VIBRATO', 'BREATH', 'BEND', 'FALL', 'TRILL', 'FALSE FINGERING', 'GROWL', 'ALTISSIMO', 'GLISSANDO']
labels = ['Vibrato', 'Breath', 'Bend', 'Fall', 'Trill', 'False Fingering', 'Growl', 'Altissimo', 'Glissando']

valores_alto = [(alto[t] == 'YES').sum() for t in tecnicas]
valores_tenor = [(tenor[t] == 'YES').sum() for t in tecnicas]

y = np.arange(len(tecnicas))
width = 0.38

axes.barh(y + width/2, valores_alto, width, label='Saxo Alto', color=ALTO_COLOR, alpha=0.85)
axes.barh(y - width/2, valores_tenor, width, label='Saxo Tenor', color=TENOR_COLOR, alpha=0.85)
axes.set_yticks(y)
axes.set_yticklabels(labels, fontsize=9)
axes.set_xlabel('Número de Grabaciones')
axes.set_title('Técnicas Extendidas')
axes.legend(loc='upper right')

plt.tight_layout()
plt.savefig("imagen3.pdf", bbox_inches='tight')
plt.savefig("imagen3.png", bbox_inches='tight')
plt.close()

# Figura 4: Distribución de tempo
fig, ax = plt.subplots(figsize=(6, 4))
bins = [75, 82, 87, 92, 97, 102, 107, 112, 117, 122]
ax.hist(alto['TEMPO'], bins=bins, alpha=0.75, label='Saxo Alto', color=ALTO_COLOR, edgecolor='white')
ax.hist(tenor['TEMPO'], bins=bins, alpha=0.75, label='Saxo Tenor', color=TENOR_COLOR, edgecolor='white')

ax.set_xlabel('Tempo (BPM)')
ax.set_ylabel('Número de Grabaciones')
ax.set_title('Distribución de Tempo')

plt.tight_layout()
plt.savefig("imagen4.pdf", bbox_inches='tight')
plt.savefig("imagen4.png", bbox_inches='tight')
plt.close()

# Figura 5: Resumen total del corpus
fig, ax = plt.subplots(figsize=(5, 3.5))

categorias = ['Total \ngrabaciones', 'Melodías', 'Ejercicios', 'Escalas']
valores = [1026, 620, 332, 74]
colores = ['#555555', ALTO_COLOR, TENOR_COLOR, '#5AAE61']
bars = ax.bar(categorias, valores, color=colores, alpha=0.85, edgecolor='white')

ax.set_ylabel('Número de Grabaciones')
ax.set_title('Resumen del Corpus (Saxofón Alto y Tenor)')

for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, str(int(bar.get_height())), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 1150)

plt.tight_layout()
plt.savefig("imagen5.pdf", bbox_inches='tight')
plt.savefig("imagen5.png", bbox_inches='tight')
plt.close()