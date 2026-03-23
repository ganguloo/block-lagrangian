import os
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_crg_log(filepath):
    init_primal = None

    db_trajectory = []
    primal_trajectory = []
    outer_ends = []
    cuts_injected = []

    re_init_primal = re.compile(r"\[Init\] Solución inicial encontrada:\s*([\-\d\.]+)")
    # NOTA: Ignoramos la extracción de init_dual.
    re_iter = re.compile(r"Iter\s+\d+:\s+Obj\s+([\-\d\.]+)\s*(\*?)\s*,\s*DB\s+([\-\d\.]+)\s*,\s*Time\s+([\d\.]+)s")
    re_heur = re.compile(r"Heuristic solution:\s*([\-\d\.]+)")
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
                    # Empezamos la trayectoria primal en t=0 con la cota inicial
                    primal_trajectory.append((0.0, current_primal_bound))

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

            m_outer = re_outer_end.search(line)
            if m_outer:
                cuts_added = int(m_outer.group(1))
                if last_seen_time > 0.0:
                    outer_ends.append(last_seen_time)
                    cuts_injected.append((last_seen_time, cuts_added))

    if primal_trajectory and last_seen_time > primal_trajectory[-1][0]:
        primal_trajectory.append((last_seen_time, current_primal_bound))

    return {
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Variables para calcular los límites globales basados en la PRIMERA ITERACIÓN
        global_max_y = -float('inf')
        global_min_y = float('inf')
        has_valid_limits = False

        colors = plt.cm.tab10.colors

        for i, (method_name, data) in enumerate(methods_data.items()):
            color = colors[i % len(colors)]

            db_times = [t for t, v in data["db_trajectory"]]
            db_vals = [v for t, v in data["db_trajectory"]]
            pr_times = [t for t, v in data["primal_trajectory"]]
            pr_vals = [v for t, v in data["primal_trajectory"]]

            # Graficar trayectorias
            ax1.plot(db_times, db_vals, label=f"{method_name} (Dual)", color=color, linewidth=2)

            if pr_times:
                ax1.step(pr_times, pr_vals, where='post', label=f"{method_name} (Primal)",
                         color=color, linestyle='--', linewidth=2)

            # Marcar fin de iteración externa
            if data["outer_ends"]:
                outer_y = []
                for ot in data["outer_ends"]:
                    closest_val = next((v for t, v in reversed(data["db_trajectory"]) if t <= ot + 1e-3), db_vals[-1])
                    outer_y.append(closest_val)
                ax1.scatter(data["outer_ends"], outer_y, color=color, marker='X', s=100, zorder=5)

            # Panel de Cortes
            if data["cuts_injected"]:
                cut_times = [t for t, c in data["cuts_injected"]]
                cut_vals = [c for t, c in data["cuts_injected"]]
                ax2.vlines(cut_times, 0, cut_vals, color=color, linewidth=3, alpha=0.7)
                ax2.plot(cut_times, cut_vals, 'o', color=color, label=f"{method_name} (Cuts)")

            # --- NUEVA LÓGICA DE LÍMITES Y ---
            if db_vals and pr_vals:
                # El techo será el primer valor Dual de la iteración 1
                local_max = db_vals[0]
                # El piso será el primer valor Primal útil (después de la inicialización)
                # O la cota inicial si es válida.
                local_min = min([v for t, v in data["primal_trajectory"] if v > -1e8] or [db_vals[-1]])

                global_max_y = max(global_max_y, local_max)
                global_min_y = min(global_min_y, local_min)
                has_valid_limits = True

        ax1.set_title(f"Convergence - {instance}", fontsize=14)
        ax1.set_ylabel("Objective", fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.7)

        # Aplicar los nuevos límites más "apretados"
        if has_valid_limits and global_min_y < global_max_y:
            margin = (global_max_y - global_min_y) * 0.05
            ax1.set_ylim(global_min_y - margin, global_max_y + margin)

        ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=9)

        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Cuts added", fontsize=12)
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
