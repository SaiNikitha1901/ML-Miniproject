import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated palette); text stays in ink colours, never series colours
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK_MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

def apply_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK_MUTED,
        'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
        'legend.frameon': False, 'legend.labelcolor': INK,
        'lines.linewidth': 2, 'lines.markersize': 7,
        'font.size': 10,
    })
