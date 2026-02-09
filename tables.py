import pandas as pd
import sys
import os

def generate_latex_report(csv_file):
    # Verificar existencia del archivo
    if not os.path.exists(csv_file):
        return f"Error: File '{csv_file}' not found."

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return f"Error reading file: {e}"

    # Columnas de agrupación y promedio
    group_cols = ['n_blocks', 'n_nodes', 'coupling', 'solver']
    avg_cols = ['total_time', 'gap', 'iter_outer', 'cuts', 'iter_inner', 'cols']

    # Mapeo de columnas para el encabezado (Header en Inglés)
    col_map = {
        'n_blocks': 'Blocks',
        'n_nodes': 'Nodes',
        'coupling': 'Coupling',
        'solver': 'Solver',
        'total_time': 'Time (s)',
        'gap': 'Gap (\%)',
        'iter_outer': 'Outer It.',
        'cuts': 'Cuts',
        'iter_inner': 'Inner It.',
        'cols': 'Columns'
    }

    # Descripción de las columnas para el itemize (en Inglés para consistencia)
    col_desc = {
        'Blocks': 'Number of blocks in the decomposition.',
        'Nodes': 'Total number of nodes in the graph.',
        'Coupling': 'Coupling parameter of the instance.',
        'Solver': 'Algorithm or solver configuration used.',
        'Time (s)': 'Average total execution time in seconds.',
        'Gap (\%)': 'Average optimality gap as a percentage.',
        'Outer It.': 'Average number of outer iterations (Master problem).',
        'Cuts': 'Average number of cuts generated.',
        'Inner It.': 'Average number of inner iterations (Pricing/Subproblems).',
        'Columns': 'Average number of columns generated.'
    }

    # --- Generación del contenido LaTeX ---
    latex_output = ""

    # 1. Generar el Itemize con descripciones
    latex_output += "% Column descriptions\n"
    latex_output += "\\section*{Column Descriptions}\n"
    latex_output += "\\begin{itemize}\n"

    # Recorremos las columnas en el orden que aparecerán (agrupación + promedio)
    ordered_cols = group_cols + avg_cols
    for col_key in ordered_cols:
        header_name = col_map.get(col_key, col_key)
        description = col_desc.get(header_name, "")
        latex_output += f"    \\item \\textbf{{{header_name}}}: {description}\n"
    latex_output += "\\end{itemize}\n\n"

    # 2. Generar las tablas por topología
    topologies = df['topo'].unique()

    for topo in topologies:
        # Filtrar y agrupar
        df_topo = df[df['topo'] == topo]
        df_grouped = df_topo.groupby(group_cols)[avg_cols].mean().reset_index()

        latex_output += f"% Table for topology: {topo}\n"
        latex_output += "\\begin{table}[ht]\n"
        latex_output += "\\small  % Small font size as requested\n"
        latex_output += "\\centering\n"
        latex_output += "\\caption{Average results for " + str(topo) + " topology}\n"

        # Definición de columnas: todas alineadas a la derecha ('r')
        num_cols = len(df_grouped.columns)
        col_alignment = "r" * num_cols
        latex_output += "\\begin{tabular}{" + col_alignment + "}\n"
        latex_output += "\\hline\n"

        # Encabezados
        headers = [col_map.get(c, c) for c in df_grouped.columns]
        latex_output += " & ".join(headers) + " \\\\\n"
        latex_output += "\\hline\n"

        # Filas de datos
        for _, row in df_grouped.iterrows():
            row_str = []
            for col in df_grouped.columns:
                val = row[col]

                # Ajuste de porcentaje para el GAP
                if col == 'gap' and not pd.isna(val):
                    val = val * 100

                # Formateo de valores
                if pd.isna(val):
                    row_str.append("-")
                elif col in ['n_blocks', 'n_nodes', 'coupling']:
                    row_str.append(str(int(val)))
                elif col == 'solver':
                    # Escapar guiones bajos para LaTeX
                    row_str.append(str(val).replace('_', r'\_'))
                elif col in ['iter_outer', 'cuts', 'iter_inner', 'cols']:
                    # 1 decimal para conteos promedio
                    row_str.append(f"{val:.1f}")
                else:
                    # Time y Gap con 2 decimales
                    row_str.append(f"{val:.2f}")

            latex_output += " & ".join(row_str) + " \\\\\n"

        latex_output += "\\hline\n"
        latex_output += "\\end{tabular}\n"
        latex_output += "\\end{table}\n\n"

    return latex_output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    result = generate_latex_report(input_file)
    print(result)
