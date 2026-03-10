import os
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_crg_effort(filepath):
    """
    Parsea un archivo log de CRG y devuelve las trayectorias de las iteraciones 
    internas acumuladas, columnas acumuladas, y tiempos de fin de iteración externa.
    """
    iter_trajectory = [] # Lista de (tiempo, n_iteracion_acumulada)
    cols_trajectory = [] # Lista de (tiempo, n_columnas_acumuladas)
    outer_ends = []      # Tiempos de fin de iteración externa
    
    # Expresiones regulares
    re_iter = re.compile(r"Iter\s+(\d+):\s+Obj.*?Time\s+([\d\.]+)s,\s*Cols\s+\+(\d+)")
    re_outer_end = re.compile(r"Fin Outer\s+\d+:")
    
    cum_cols = 0
    last_seen_time = 0.0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Buscar línea de iteración interna
            m_iter = re_iter.search(line)
            if m_iter:
                iter_num = int(m_iter.group(1))
                current_time = float(m_iter.group(2))
                added_cols = int(m_iter.group(3))
                
                cum_cols += added_cols
                
                iter_trajectory.append((current_time, iter_num))
                cols_trajectory.append((current_time, cum_cols))
                
                last_seen_time = current_time
            
            # 2. Buscar fin de iteración externa
            if re_outer_end.search(line):
                if last_seen_time > 0.0:
                    outer_ends.append(last_seen_time)
                    
    return {
        "iter_trajectory": iter_trajectory,
        "cols_trajectory": cols_trajectory,
        "outer_ends": outer_ends
    }

def main():
    parser = argparse.ArgumentParser(description="Graficar esfuerzo de CRG (Iteraciones y Columnas) desde logs")
    parser.add_argument("--logdir", type=str, default="logs", help="Directorio de logs")
    parser.add_argument("--outdir", type=str, default="plots", help="Carpeta para guardar gráficos")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # data_by_instance[instancia][metodo] = datos_parseados
    data_by_instance = defaultdict(dict)
    
    # 1. Recopilar y parsear archivos
    for filename in os.listdir(args.logdir):
        if not filename.endswith(".log") or "_CRG_" not in filename:
            continue
            
        filepath = os.path.join(args.logdir, filename)
        
        parts = filename.replace(".log", "").split("_CRG_")
        if len(parts) != 2:
            continue
            
        instance_name = parts[0]
        method_name = "CRG_" + parts[1]
        
        parsed_data = parse_crg_effort(filepath)
        if parsed_data["iter_trajectory"]: # Solo guardar si procesó iteraciones reales
            data_by_instance[instance_name][method_name] = parsed_data

    if not data_by_instance:
        print(f"No se encontraron logs de CRG con datos válidos en {args.logdir}/")
        return

    # 2. Generar un gráfico por instancia
    for instance, methods_data in data_by_instance.items():
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Crear un segundo eje Y compartiendo el mismo eje X
        ax2 = ax1.twinx()
        
        colors = plt.cm.tab10.colors
        
        for i, (method_name, data) in enumerate(methods_data.items()):
            color = colors[i % len(colors)]
            
            # Extraer trayectorias
            iter_t = [t for t, v in data["iter_trajectory"]]
            iter_v = [v for t, v in data["iter_trajectory"]]
            
            cols_t = [t for t, v in data["cols_trajectory"]]
            cols_v = [v for t, v in data["cols_trajectory"]]
            
            # Plot en Eje Y Izquierdo (Iteraciones) - Línea Sólida
            ax1.plot(iter_t, iter_v, label=f"{method_name} (Iters)", color=color, linestyle='-', linewidth=2)
            
            # Plot en Eje Y Derecho (Columnas) - Línea Punteada
            ax2.plot(cols_t, cols_v, label=f"{method_name} (Cols)", color=color, linestyle='--', linewidth=2)
                
            # Marcar fines de iteración externa en AMBAS curvas
            if data["outer_ends"]:
                outer_iter_y = []
                outer_cols_y = []
                for ot in data["outer_ends"]:
                    # Buscar el valor Y más cercano a este tiempo (hacia atrás)
                    val_iter = next((v for t, v in reversed(data["iter_trajectory"]) if t <= ot + 1e-3), iter_v[-1])
                    val_cols = next((v for t, v in reversed(data["cols_trajectory"]) if t <= ot + 1e-3), cols_v[-1])
                    
                    outer_iter_y.append(val_iter)
                    outer_cols_y.append(val_cols)
                
                # Pintar la X en la curva de iteraciones (ax1)
                ax1.scatter(data["outer_ends"], outer_iter_y, color=color, marker='X', s=100, zorder=5)
                # Pintar la X en la curva de columnas (ax2)
                ax2.scatter(data["outer_ends"], outer_cols_y, color=color, marker='X', s=100, zorder=5)

        # Configuración del gráfico
        plt.title(f"Esfuerzo Computacional CRG - {instance}", fontsize=14)
        ax1.set_xlabel("Tiempo (segundos)", fontsize=12)
        
        ax1.set_ylabel("Cantidad de Iteraciones Internas", fontsize=12)
        ax2.set_ylabel("Columnas Totales Agregadas", fontsize=12)
        
        # Grid alineado solo al eje primario para no saturar visualmente
        ax1.grid(True, linestyle=':', alpha=0.7)
        
        # Juntar leyendas de ambos ejes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.10, 1), fontsize=9)
        
        # Evitar que la leyenda quede cortada
        fig.tight_layout()
        
        out_path = os.path.join(args.outdir, f"effort_{instance}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de esfuerzo guardado: {out_path}")

if __name__ == "__main__":
    main()