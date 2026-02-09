
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

@dataclass
class EdgeInfo:
    u: int
    v: int
    vars_u: List[int]
    vars_v: List[int]

class TopologyManager:
    def __init__(self, block_sizes: Union[int, List[int]]):
        """
        Args:
            block_sizes: Puede ser una lista de enteros donde block_sizes[i] es
                         el nro de variables del bloque i, o un entero si todos
                         tienen el mismo tamaño (se asume n_blocks inferido luego
                         o no aplicable en modo entero único, mejor pasar lista).
        """
        if isinstance(block_sizes, int):
            # Modo legacy/simple: se asume que el usuario definirá la topología
            # y el número de bloques implícitamente, o esto es solo un placeholder.
            # Para robustez, asumimos que es una lista si queremos heterogeneidad.
            # Aquí guardamos el tamaño default.
            self.block_sizes = {} # Se llenará bajo demanda o error
            self.default_size = block_sizes
            self.num_blocks = 0
        else:
            self.block_sizes = block_sizes
            self.num_blocks = len(block_sizes)
            self.default_size = None

        self.edges: Dict[Tuple[int, int], EdgeInfo] = {}
        self.adj: Dict[int, List[int]] = {i: [] for i in range(self.num_blocks)}

    def _get_size(self, block_id: int) -> int:
        if isinstance(self.block_sizes, list):
            return self.block_sizes[block_id]
        return self.default_size

    def add_coupling(self, u: int, v: int, indices_u: List[int], indices_v: List[int]):
        key = tuple(sorted((u, v)))
        self.edges[key] = EdgeInfo(u, v, indices_u, indices_v)
        if u not in self.adj: self.adj[u] = []
        if v not in self.adj: self.adj[v] = []
        self.adj[u].append(v)
        self.adj[v].append(u)

    def get_edge(self, u: int, v: int) -> EdgeInfo:
        return self.edges.get(tuple(sorted((u, v))))

    def get_neighbors(self, u: int) -> List[int]:
        return self.adj.get(u, [])

    def create_star(self, center: int, coupling_size: int):
        """
        Estrella: El centro comparte sus primeras variables con el inicio de cada hoja.
        Center[0...C] <==> Leaf[0...C]
        (Sincronización masiva de las primeras variables)
        """
        center_size = self._get_size(center)
        if center_size < coupling_size:
            raise ValueError(f"Block {center} size {center_size} < coupling {coupling_size}")

        indices_center = list(range(coupling_size))

        for i in range(self.num_blocks):
            if i != center:
                leaf_size = self._get_size(i)
                if leaf_size < coupling_size:
                    raise ValueError(f"Block {i} size {leaf_size} < coupling {coupling_size}")

                indices_leaf = list(range(coupling_size))
                self.add_coupling(center, i, indices_center, indices_leaf)

    def create_path(self, coupling_size: int):
        """
        Camino: Block[i] (Fin) <==> Block[i+1] (Inicio)
        """
        for i in range(self.num_blocks - 1):
            u, v = i, i+1
            size_u = self._get_size(u)
            size_v = self._get_size(v)

            if size_u < coupling_size or size_v < coupling_size:
                raise ValueError(f"Blocks {u},{v} too small for coupling {coupling_size}")

            # Salida de U: Últimas variables
            indices_u = list(range(size_u - coupling_size, size_u))
            # Entrada de V: Primeras variables
            indices_v = list(range(coupling_size))

            self.add_coupling(u, v, indices_u, indices_v)

    def create_bintree(self, coupling_size: int):
        """
        Arbol Binario (Estocástico):
        Padre (Fin) <==> Hijo Izq (Inicio)
        Padre (Fin) <==> Hijo Der (Inicio)
        El padre comparte el MISMO estado (sus últimas variables) con ambos hijos.
        """
        for i in range(self.num_blocks):
            size_parent = self._get_size(i)

            # Puerto de salida del padre (Cola)
            if size_parent < coupling_size:
                 raise ValueError(f"Block {i} too small")
            indices_u = list(range(size_parent - coupling_size, size_parent))

            # Hijos
            children = []
            if 2 * i + 1 < self.num_blocks: children.append(2 * i + 1)
            if 2 * i + 2 < self.num_blocks: children.append(2 * i + 2)

            for child in children:
                size_child = self._get_size(child)
                if size_child < coupling_size:
                    raise ValueError(f"Child Block {child} too small")

                # Puerto de entrada del hijo (Cabeza)
                indices_v = list(range(coupling_size))
                self.add_coupling(i, child, indices_u, indices_v)
