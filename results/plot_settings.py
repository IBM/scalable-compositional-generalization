import matplotlib
# matplotlib.use('pgf')

def set_latex_settings():
    # Use the pgf backend
    matplotlib.pyplot.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,            # Use LaTeX for text rendering
        "font.family": "serif",         # Use a serif font (matches LaTeX default)
        "font.serif": ["Times New Roman"], # Computer Modern Roman font (Neurips uses Times New Roman)
        "font.size": 10,                # Match LaTeX document font size
        "axes.labelsize": 10,           # Label size
        "axes.titlesize": 10,           # Title size
        "legend.fontsize": 9,           # Legend font size
        "xtick.labelsize": 9,           # X-axis tick size
        "ytick.labelsize": 9,           # Y-axis tick size
        "text.latex.preamble": r"\usepackage{amsmath, amssymb}",  # Custom LaTeX packages
        "pgf.rcfonts": False,           # Don't override rc settings with default pgf settings
    })

column_width = 5.5 # Neurips template

def format(setting_name:str):
    color =         {'generate \& evaluate': 'k',
                    'generate all \& TS': 'b',
                    'TSFT': 'r',
                    'TSFT-2': 'tab:orange',
                    'TSFT-4': 'tab:green',
                    'TSFT-8': 'tab:purple',
                    'TSFT-16': 'tab:brown',
                    'TSFT-32': 'tab:pink',
                    'TSFT-64': 'r' 
                    }
    line_style =    {'generate \& evaluate': '-',
                    'generate all \& TS': '--',
                    'TSFT': ':',
                    'TSFT-2': ':',
                    'TSFT-4': ':',
                    'TSFT-8': ':',
                    'TSFT-16': ':',
                    'TSFT-32': ':',
                    'TSFT-64': ':' 
                    }
    return {'color': color[setting_name], 'linestyle': line_style[setting_name]}
