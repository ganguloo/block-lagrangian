from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import gurobipy as gp

class SeparationStrategy(ABC):
    """
    Clase base abstracta para estrategias de separación (Corte).
    Define la interfaz que CRGManager espera.
    """

    def get_w_signature(self, x_values: List[int]) -> Tuple[int, ...]:
        """
        Genera una firma hashable (tupla) que representa la configuración de frontera.
        Por defecto, es la tupla densa de valores (0, 1, 0, ...).
        Las estrategias pueden sobrescribir esto si necesitan representaciones dispersas.
        """
        return tuple(x_values)

    @abstractmethod
    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Any]:
        """
        Identifica violaciones entre las distribuciones de probabilidad de dos bloques.
        Retorna una lista de 'cortes' (pueden ser firmas, máscaras, o objetos custom).
        """
        pass

    @abstractmethod
    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var], 
                              cuts: List[Any], duals: Dict) -> gp.LinExpr:
        """
        Construye la expresión de penalización para la función objetivo del subproblema (Pricing).
        """
        pass
    
    @abstractmethod
    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        """
        Retorna el coeficiente de una columna en un corte dado.
        Usado por el Maestro para construir la restricción lineal.
        """
        pass