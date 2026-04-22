import gurobipy as gp
from typing import List, Dict, Tuple, Any
from .base_strategy import SeparationStrategy
from .exact_m_lagrangian import ExactMLagrangianStrategy
from .v_lagrangian import VLagrangianStrategy

class HybridMLagrangianStrategy(SeparationStrategy):
    def __init__(self, 
                 exact_tolerance: float = 1e-6, exact_factor: float = 0.5,
                 vlag_radius_factor: float = 0.0, vlag_tolerance: float = 1e-6, vlag_factor: float = 1.0,
                 max_outer_iters: int = None, 
                 max_time: float = None, 
                 max_cuts: int = None,
                 single_threaded: bool = False):
        
        super().__init__(single_threaded=single_threaded)
        
        # Instanciar las dos estrategias subyacentes
        self.exact_strategy = ExactMLagrangianStrategy(tolerance=exact_tolerance, factor=exact_factor, single_threaded=single_threaded)
        self.vlag_strategy = VLagrangianStrategy(radius_factor=vlag_radius_factor, tolerance=vlag_tolerance, factor=vlag_factor, single_threaded=single_threaded)
        
        # Límites de transición (None significa que no aplica)
        self.max_outer_iters = max_outer_iters
        self.max_time = max_time
        self.max_cuts = max_cuts
        
        # Estado inicial
        self.current_mode = "exact"

    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Tuple]:
        """
        Delega la separación a la estrategia activa y envuelve el resultado.
        """
        if self.current_mode == "exact":
            raw_cuts = self.exact_strategy.separate(w_sol_u, w_sol_v)
            # Etiquetamos el corte para saber de dónde vino
            return [("exact", sig) for sig in raw_cuts]
        else:
            raw_cuts = self.vlag_strategy.separate(w_sol_u, w_sol_v)
            return [("vlag", sig) for sig in raw_cuts]

    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        """
        Enruta la evaluación dependiendo de la etiqueta del corte.
        """
        tag, real_sig = cut_signature
        
        if tag == "exact":
            return self.exact_strategy.evaluate_cut(column_signature, real_sig)
        elif tag == "vlag":
            return self.vlag_strategy.evaluate_cut(column_signature, real_sig)
        
        return 0.0

    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        """
        Desempaqueta los cortes y los envía a sus respectivos constructores de penalización.
        """
        exact_cuts = []
        vlag_cuts = []
        
        for cut_id, wrapper_sig, sign_factor in cuts:
            tag, real_sig = wrapper_sig
            if tag == "exact":
                exact_cuts.append((cut_id, real_sig, sign_factor))
            elif tag == "vlag":
                vlag_cuts.append((cut_id, real_sig, sign_factor))
                
        total_penalty = gp.LinExpr()
        
        # Sumar las penalizaciones de ExactM (si hay cortes activos de esa fase)
        if exact_cuts:
            total_penalty.add(self.exact_strategy.apply_pricing_penalty(model, vars_list, exact_cuts, duals))
            
        # Sumar las penalizaciones de VLag (si hay cortes activos de esta fase)
        if vlag_cuts:
            total_penalty.add(self.vlag_strategy.apply_pricing_penalty(model, vars_list, vlag_cuts, duals))
            
        return total_penalty

    def update_state(self, current_iter: int, current_time: float, total_cuts: int):
        """
        Verifica los límites para hacer el cambio definitivo a VLag.
        """
        if self.current_mode == "exact":
            switch = False
            reasons = []
            
            if self.max_outer_iters is not None and current_iter >= self.max_outer_iters:
                switch = True
                reasons.append(f"Iteraciones ({current_iter} >= {self.max_outer_iters})")
                
            if self.max_time is not None and current_time >= self.max_time:
                switch = True
                reasons.append(f"Tiempo ({current_time:.1f}s >= {self.max_time}s)")
                
            if self.max_cuts is not None and total_cuts >= self.max_cuts:
                switch = True
                reasons.append(f"Cortes ({total_cuts} >= {self.max_cuts})")
                
            if switch:
                self.current_mode = "vlag"
                print(f"\\n  >>> [Híbrido] Cambiando a VLagrangianStrategy. Motivo: {', '.join(reasons)}")