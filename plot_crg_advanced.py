import os
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_crg_log(filepath):
    init_primal = None
    init_dual = None
    
    db_trajectory = []     
    primal_trajectory = [] 
    outer_ends = []        
    cuts_injected = []     # NUEVO: Lista de tuplas (tiempo, cantidad_de_cortes)
    
    re_init_primal = re.compile(r"\[Init\] Solución inicial encontrada:\s*([\-\d\.]+)")
    re_init_dual = re.compile(r"\[Init\] Cota Dual Inicial \(LR\):\s*([\-\d\.]+)")
    re_iter = re.compile(r"Iter\s+\d+:\s+Obj\s+([\-\d\.]+)\s*(\*?)\s*,\s*DB\s+([\-\d\.]+)\s*,\s*Time\s+([\d\.]+)s")
    re_heur = re.compile(r"Heuristic solution:\s*([\-\d\.]+)")
    # NUEVO: Extraer la cantidad de cortes de la línea "Fin Outer"
    re_outer_end = re.compile(r"Fin Outer\s+\d+.*?Cuts\s+\+(\d+)")
    
    last_seen_time = 0.0
    current_primal_bound = -float('inf')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m_primal = re_init_primal.search(line)
            if m_primal:
                init_primal = float(m_primal.group(1))
                if init_primal > -1e8:
                    current_primal_bound = init_primal
                    primal_trajectory.append((0.0, current_primal_bound))
                
            m_dual = re_init_dual.search(line)
            if m_dual:
                init_dual = float(m_dual.group(1))
                
            m_iter = re_iter.search(line)
            if m_iter:
                obj_val = float(m_iter.group(1))
                has_asterisk = m_iter.group(2) == '*'
                db_val = float(m_iter.group(3))
                current_time = float(m_iter.group(4))
                
                db_trajectory.append((current_time, db_val))
                last_seen_time = current_time
                
                if has_asterisk and obj_val > current_primal_bound:
                    current_primal_bound = obj_val
                    primal_trajectory.append((current_time, current_primal_bound))
            
            m_heur = re_heur.search(line)
            if m_heur:
                heur_val = float(m_heur.group(1))
                if heur_val > current_primal_bound:
                    current_primal_bound = heur_val
                    primal_trajectory.append((last_seen_time, current_primal_bound))
            
            # NUEVO: Parsear la inyección de cortes
            m_outer = re_outer_end.search(line)
            if m_outer:
                cuts_added = int(m_outer.group(1))
                if last_seen_time > 0.0:
                    outer_ends.append(last_seen_time)
                    cuts_injected.append((last_seen_time, cuts_added))
                    
    if primal_trajectory and last_seen_time > primal_trajectory[-1][0]:
        primal_trajectory.append((last_seen_time, current_primal_bound))
        
    return {
        "init_primal": init_primal,
        "init_dual": init_dual,
        "db_trajectory": db_trajectory,
        "primal_trajectory": primal_trajectory,
        "outer_ends": outer_ends,
        "cuts_injected": cuts_injected
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data_by_instance = defaultdict(dict)
    
    for filename in os.listdir(args.logdir):
        if not filename.endswith(".log") or "_CRG_" not in filename: continue
        filepath = os.path.join(args.logdir, filename)
        parts = filename.replace(".log", "").split("_CRG_")
        if len(parts) != 2: continue
        
        instance_name = parts[0]
        method_name = "CRG_" + parts[1]
        
        parsed_data = parse_crg_log(filepath)
        if parsed_data["db_trajectory"]:
            data_by_instance[instance_name][method_name] = parsed_data

    for instance, methods_data in data_by_instance.items():
        # Crear 2 subplots apilados compartiendo el eje X
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        global_init_primal = float('inf')
        global_init_dual = -float('inf')
        has_valid_limits = False
        colors = plt.cm.tab10.colors
        
        for i, (method_name, data) in enumerate(methods_data.items()):
            color = colors[i % len(colors)]
            
            # --- PANEL 1: CONVERGENCIA ---
            db_times = [t for t, v in data["db_trajectory"]]
            db_vals = [v for t, v in data["db_trajectory"]]
            pr_times = [t for t, v in data["primal_trajectory"]]
            pr_vals = [v for t, v in data["primal_trajectory"]]
            
            ax1.plot(db_times, db_vals, label=f"{method_name} (Dual)", color=color, linewidth=2)
            
            if pr_times:
                ax1.step(pr_times, pr_vals, where='post', label=f"{method_name} (Primal)", 
                         color=color, linestyle='--', linewidth=2)
                
            if data["outer_ends"]:
                outer_y = []
                for ot in data["outer_ends"]:
                    closest_val = next((v for t, v in reversed(data["db_trajectory"]) if t <= ot + 1e-3), db_vals[-1])
                    outer_y.append(closest_val)
                ax1.scatter(data["outer_ends"], outer_y, color=color, marker='X', s=100, zorder=5)

            # --- PANEL 2: CORTES AGREGADOS ---
            if data["cuts_injected"]:
                cut_times = [t for t, c in data["cuts_injected"]]
                cut_vals = [c for t, c in data["cuts_injected"]]
                
                # Dibujar impulsos (líneas verticales con un punto arriba)
                ax2.vlines(cut_times, 0, cut_vals, color=color, linewidth=3, alpha=0.7)
                ax2.plot(cut_times, cut_vals, 'o', color=color, label=f"{method_name} (Cortes)")

            # Rango Y
            if data["primal_trajectory"] and data["primal_trajectory"][0][1] > -1e8:
                global_init_primal = min(global_init_primal, data["primal_trajectory"][0][1])
                has_valid_limits = True
            if data["init_dual"] is not None and data["init_dual"] < 1e8:
                global_init_dual = max(global_init_dual, data["init_dual"])
                has_valid_limits = True

        # Formato Panel Superior (Convergencia)
        ax1.set_title(f"Convergencia y Cortes Lagrangianos - {instance}", fontsize=14)
        ax1.set_ylabel("Valor Objetivo", fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.7)
        if has_valid_limits and global_init_primal < global_init_dual:
            margin = (global_init_dual - global_init_primal) * 0.05
            ax1.set_ylim(global_init_primal - margin, global_init_dual + margin)
        ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=9)

        # Formato Panel Inferior (Cortes)
        ax2.set_xlabel("Tiempo (segundos)", fontsize=12)
        ax2.set_ylabel("Cortes Inyectados", fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.7, axis='y')
        ax2.set_ylim(bottom=0)
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        out_path = os.path.join(args.outdir, f"plot_adv_{instance}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Gráfico avanzado guardado: {out_path}")

if __name__ == "__main__":
    main()