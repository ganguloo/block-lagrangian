import os
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_crg_log(filepath):
    """
    Parsea un archivo log de CRG y devuelve las trayectorias de Dual Bound,
    Primal Bound, los tiempos de fin de iteración externa y los límites iniciales.
    """
    init_primal = None
    init_dual = None
    
    db_trajectory = []     
    primal_trajectory = [] 
    outer_ends = []        
    
    # Expresiones regulares
    re_init_primal = re.compile(r"\[Init\] Solución inicial encontrada:\s*([\-\d\.]+)")
    re_init_dual = re.compile(r"\[Init\] Cota Dual Inicial \(LR\):\s*([\-\d\.]+)")
    re_iter = re.compile(r"Iter\s+\d+:\s+Obj\s+([\-\d\.]+)\s*(\*?)\s*,\s*DB\s+([\-\d\.]+)\s*,\s*Time\s+([\d\.]+)s")
    re_heur = re.compile(r"Heuristic solution:\s*([\-\d\.]+)")
    re_outer_end = re.compile(r"Fin Outer\s+\d+")
    
    last_seen_time = 0.0
    current_primal_bound = -float('inf')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Buscar cotas iniciales
            m_primal = re_init_primal.search(line)
            if m_primal:
                init_primal = float(m_primal.group(1))
                if init_primal > -1e8:  # Evitar heurísticos fallidos (-1e9)
                    current_primal_bound = init_primal
                    primal_trajectory.append((0.0, current_primal_bound))
                
            m_dual = re_init_dual.search(line)
            if m_dual:
                init_dual = float(m_dual.group(1))
                
            # 2. Buscar iteraciones internas (Pricing y Maestro)
            m_iter = re_iter.search(line)
            if m_iter:
                obj_val = float(m_iter.group(1))
                has_asterisk = m_iter.group(2) == '*'
                db_val = float(m_iter.group(3))
                current_time = float(m_iter.group(4))
                
                db_trajectory.append((current_time, db_val))
                last_seen_time = current_time
                
                # Actualizar incumbente si el maestro encontró una solución entera mejor
                if has_asterisk and obj_val > current_primal_bound:
                    current_primal_bound = obj_val
                    primal_trajectory.append((current_time, current_primal_bound))
            
            # 3. Buscar MIP Heurístico al final de iteración externa
            m_heur = re_heur.search(line)
            if m_heur:
                heur_val = float(m_heur.group(1))
                # Actualizar incumbente si el heurístico encontró algo mejor
                if heur_val > current_primal_bound:
                    current_primal_bound = heur_val
                    # Usamos el last_seen_time de la iteración inmediatamente anterior
                    primal_trajectory.append((last_seen_time, current_primal_bound))
            
            # 4. Buscar fin de iteración externa para las cruces (X)
            if re_outer_end.search(line):
                if last_seen_time > 0.0:
                    outer_ends.append(last_seen_time)
                    
    # FORZAR EXTENSIÓN: Añadir un punto final en el último tiempo registrado para que
    # la línea escalonada abarque todo el eje X de principio a fin.
    if primal_trajectory and last_seen_time > primal_trajectory[-1][0]:
        primal_trajectory.append((last_seen_time, current_primal_bound))
        
    return {
        "init_primal": init_primal,
        "init_dual": init_dual,
        "db_trajectory": db_trajectory,
        "primal_trajectory": primal_trajectory,
        "outer_ends": outer_ends
    }

def main():
    parser = argparse.ArgumentParser(description="Graficar resultados de CRG desde logs")
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
        
        # Asumimos formato: {instancia}_{Metodo}.log
        parts = filename.replace(".log", "").split("_CRG_")
        if len(parts) != 2:
            continue
            
        instance_name = parts[0]
        method_name = "CRG_" + parts[1]
        
        parsed_data = parse_crg_log(filepath)
        if parsed_data["db_trajectory"]: # Solo procesar si corrió al menos una iteración
            data_by_instance[instance_name][method_name] = parsed_data

    if not data_by_instance:
        print(f"No se encontraron logs de CRG con datos válidos en {args.logdir}/")
        return

    # 2. Generar un gráfico por instancia
    for instance, methods_data in data_by_instance.items():
        plt.figure(figsize=(12, 7))
        
        # Límites globales para asegurar que todas las curvas de esta instancia encajen
        global_init_primal = float('inf')
        global_init_dual = -float('inf')
        has_valid_limits = False
        
        colors = plt.cm.tab10.colors # Paleta de 10 colores
        
        for i, (method_name, data) in enumerate(methods_data.items()):
            color = colors[i % len(colors)]
            
            # Extraer trayectorias
            db_times = [t for t, v in data["db_trajectory"]]
            db_vals = [v for t, v in data["db_trajectory"]]
            
            pr_times = [t for t, v in data["primal_trajectory"]]
            pr_vals = [v for t, v in data["primal_trajectory"]]
            
            # Línea continua de Dual Bound
            plt.plot(db_times, db_vals, label=f"{method_name} (Dual)", color=color, linewidth=2)
            
            # Línea escalonada de Primal Bound extendida
            if pr_times:
                plt.step(pr_times, pr_vals, where='post', label=f"{method_name} (Primal)", 
                         color=color, linestyle='--', linewidth=2)
                
            # Marcar fines de iteración externa con una X gigante
            if data["outer_ends"]:
                outer_y = []
                for ot in data["outer_ends"]:
                    # Buscar el valor de la cota dual más cercano a ese tiempo
                    closest_val = next((v for t, v in reversed(data["db_trajectory"]) if t <= ot + 1e-3), db_vals[-1])
                    outer_y.append(closest_val)
                
                plt.scatter(data["outer_ends"], outer_y, color=color, marker='X', s=100, zorder=5, 
                            label=f"{method_name} (Fin Outer)" if i == 0 else "")

            # Calcular el rango para el Eje Y
            if data["primal_trajectory"] and data["primal_trajectory"][0][1] > -1e8:
                global_init_primal = min(global_init_primal, data["primal_trajectory"][0][1])
                has_valid_limits = True
            if data["init_dual"] is not None and data["init_dual"] < 1e8:
                global_init_dual = max(global_init_dual, data["init_dual"])
                has_valid_limits = True

        # Títulos y formato del gráfico
        plt.title(f"Convergencia Lagrangiana - {instance}", fontsize=14)
        plt.xlabel("Tiempo (segundos)", fontsize=12)
        plt.ylabel("Valor Objetivo", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Aplicar los límites de corte en el eje Y
        if has_valid_limits and global_init_primal < global_init_dual:
            margin = (global_init_dual - global_init_primal) * 0.05
            plt.ylim(global_init_primal - margin, global_init_dual + margin)
            
        plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=9)
        plt.tight_layout()
        
        # Guardar en disco
        out_path = os.path.join(args.outdir, f"plot_{instance}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Gráfico guardado: {out_path}")

if __name__ == "__main__":
    main()