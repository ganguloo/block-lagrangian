
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import gurobipy as gp

class SeparationStrategy(ABC):
    @abstractmethod
    def get_w_signature(self, x_values: List[int]) -> Tuple:
        pass

    @abstractmethod
    def separate(self, w_sol_u: Dict[Tuple, float], w_sol_v: Dict[Tuple, float]) -> List[Any]:
        pass

    @abstractmethod
    def apply_pricing_penalty(self, model: gp.Model, vars_list: List[gp.Var],
                              cuts: List[Tuple], duals: Dict):
        pass

    @abstractmethod
    def evaluate_cut(self, column_signature: Tuple, cut_signature: Any) -> float:
        pass
