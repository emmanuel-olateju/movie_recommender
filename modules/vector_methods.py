import numpy as np
from typing import List, Union
from search import binary_search


class MatrixEntry:

    def __init__(self, value: float, idx: int) -> None:
        self.value = value
        self.col_idx = idx

    def __repr__(self) -> str:
        return f"MatrixEntry(value={self.value}, col_idx={self.col_idx})"

class DictVector:

    def __init__(self, vector: List[Union[int, float, np.ndarray, list, 'DictVector']], idx: int):
        if not vector:
            raise ValueError("DictVector cannot be empty")
        
        self.vector = {}
        self.vidx = self.vector_idx = idx
        self.item_type = type(vector[0])
        self.length = len(vector)

        for j, entry in enumerate(vector):
            assert type(entry) is self.item_type

            if isinstance(entry, np.ndarray) or isinstance(entry, list):
                if isinstance(entry, list):
                    nested_row = DictVector(entry, j)
                elif isinstance(entry, np.ndarray):
                    nested_row = DictVector(entry.tolist(), j)
                    
                if len(nested_row) > 0:
                    self.vector[j] = nested_row
            elif (entry != 0) and (entry != 0.0):
                self.vector[j] = entry
            else:
                pass

    def __getitem__(self, index: int) -> Union['MatrixEntry', int, float]:
        if index >= self.length:
            raise IndexError(f"Index {index} out of bounds for vecotr of size {self.length}")
        
        result = self.vector.get(index)
        if result is None:
            if self.item_type is int:
                return 0
            elif self.item_type is float:
                return 0.0
            else:
                return None
        
        return result
    
    def __len__(self) -> int:
        return len(self.vector)
    
    def __repr__(self):
        return f"DictVector(idx={self.vector_idx}), stored_entries={self.vector}, values={dict(self.vector)}"


class ListVector:

    def __init__(self, vector: List[Union[int, float, np.ndarray, list, 'ListVector']], idx: int):
        self.vector = []
        self.vidx = []
        self.idx = idx
        self.item_type = type(vector[0])
        self.length = len(vector)

        for j, entry in enumerate(vector):
            assert type(entry) is self.item_type

            if isinstance(entry, np.ndarray) or isinstance(entry, list):
                if isinstance(entry, list):
                    nested_vector = ListVector(entry,  j)
                elif isinstance(entry, np.ndarray):
                    nested_vector = ListVector(entry.tolist(), j)
                
                if len(nested_vector) > 0:
                    self.vector.append(nested_vector)
                    self.vidx.append(j)
            elif (entry != 0) and (entry != 0.0):
                self.vector.append(entry)
                self.vidx.append(j)
            else:
                pass

    def __getitem__(self, index: Union[int]) -> Union[int, float, 'ListVector']:
        if index >= self.length:
            raise IndexError(f"Index {index} out of bounds for vecotr of size {self.length}")
        
        idx = binary_search(self.vidx, index)
        if idx is None:
            if self.item_type is int:
                return 0
            elif self.item_type is float:
                return 0.0
            else:
                return None
            
        return self.vector[idx]
    
    def __len__(self) -> int:
        return len(self.vector)
    
    def __repr__(self):
        return f"ListVector(idx={self.vidx}), stored_entries={self.vector}"