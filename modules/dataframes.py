import numpy as np
from typing import List, Union
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))


from sparse_matrix import RowVector, MatrixMthd_1


class Dataframe_Mthd1:

    def __init__(self, matrix: Union['MatrixMthd_1', np.ndarray], index: List[Union[int, str]], columns: List[Union[int, str]]):
        self.index = {idx: i for i, idx in enumerate(index)}
        self.columns = {col: i for i, col in enumerate(columns)}

        if isinstance(matrix, np.ndarray):
            self.matrix = MatrixMthd_1(matrix)
        elif isinstance(matrix, 'MatrixMthd_1'):
            self.matrix = matrix

    
    def __getitem__(self, index: Union[int, List[int]]) -> Union['RowVector', int, float]:
        if len(index) == 1:
            idx = self.index[index[0]]
        elif len(index) == 2:
            idx = self.index[index[0]]
            col = self.columns[index[1]]

        return self.matrix[idx, col]