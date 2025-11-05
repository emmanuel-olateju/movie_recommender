import numpy as np
from typing import List, Union
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))
from vector_methods import DictVector, ListVector
from sparse_matrix import MatrixMthd_1, MatrixMthd_2

class Dataframe_Mthd1:
    def __init__(self, matrix: Union['MatrixMthd_1', np.ndarray], index: List[Union[int, str]], columns: List[Union[int, str]]):
        self.index = {idx: i for i, idx in enumerate(index)}
        self.columns = {col: i for i, col in enumerate(columns)}
        if isinstance(matrix, np.ndarray):
            self.matrix = MatrixMthd_1(matrix)
        elif isinstance(matrix, MatrixMthd_1):
            self.matrix = matrix
    
    def __getitem__(self, index: Union[int, List[int]]) -> Union['DictVector', int, float]:
        if len(index) == 1:
            idx = self.index[index[0]]
            return self.matrix[idx]
        elif len(index) == 2:
            idx = self.index[index[0]]
            col = self.columns[index[1]]
            return self.matrix[idx, col]

class Dataframe_Mthd2:
    def __init__(self, matrix: Union['MatrixMthd_2', np.ndarray], index: List[Union[int, str]], columns: List[Union[int, str]]):
        self.index = {idx: i for i, idx in enumerate(index)}
        self.columns = {col: i for i, col in enumerate(columns)}
        if isinstance(matrix, np.ndarray):
            self.matrix = MatrixMthd_2(matrix)
        elif isinstance(matrix, MatrixMthd_2):
            self.matrix = matrix
    
    def __getitem__(self, index: Union[int, List[int]]) -> Union['ListVector', int, float]:
        if len(index) == 1:
            idx = self.index[index[0]]
            return self.matrix[idx]
        elif len(index) == 2:
            idx = self.index[index[0]]
            col = self.columns[index[1]]
            return self.matrix[idx, col]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Dataframe with Sparse Matrices (MatrixMthd_2)")
    print("=" * 60)
    
    # Test 1: Create a sparse matrix (mostly zeros)
    print("\n--- Test 1: Creating sparse dataframe ---")
    sparse_data = np.array([
        [0, 0, 3, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 5],
        [1, 0, 0, 4, 0]
    ], dtype=float)
    
    index_labels = ['row1', 'row2', 'row3', 'row4']
    column_labels = ['A', 'B', 'C', 'D', 'E']
    
    df_sparse = Dataframe_Mthd2(sparse_data, index_labels, column_labels)
    print(f"Created sparse dataframe with shape: {sparse_data.shape}")
    print(f"Sparsity: {np.sum(sparse_data == 0) / sparse_data.size * 100:.1f}% zeros")
    
    # Test 2: Access individual elements
    print("\n--- Test 2: Accessing individual elements ---")
    test_cases = [
        (['row1', 'C'], 3.0),
        (['row2', 'B'], 2.0),
        (['row3', 'E'], 5.0),
        (['row4', 'A'], 1.0),
        (['row1', 'A'], 0.0),  # Zero element
        (['row2', 'E'], 0.0),  # Zero element
    ]
    
    for indices, expected in test_cases:
        result = df_sparse[indices]
        status = "✓" if result == expected else "✗"
        print(f"{status} df_sparse[{indices}] = {result} (expected: {expected})")
    
    # Test 3: Access entire rows
    print("\n--- Test 3: Accessing rows ---")
    row1 = df_sparse[['row1']]
    print(f"Row 'row1': {row1}")
    
    row4 = df_sparse[['row4']]
    print(f"Row 'row4': {row4}")
    
    # Test 4: Very sparse matrix (high sparsity)
    print("\n--- Test 4: High sparsity matrix (95% zeros) ---")
    large_sparse = np.zeros((10, 10))
    large_sparse[0, 5] = 10
    large_sparse[3, 2] = 7
    large_sparse[7, 8] = 3
    large_sparse[9, 1] = 15
    large_sparse[5, 5] = 20
    
    idx = [f'r{i}' for i in range(10)]
    cols = [f'c{i}' for i in range(10)]
    
    df_large = Dataframe_Mthd2(large_sparse, idx, cols)
    print(f"Created large sparse dataframe: {large_sparse.shape}")
    print(f"Sparsity: {np.sum(large_sparse == 0) / large_sparse.size * 100:.1f}% zeros")
    
    non_zero_tests = [
        (['r0', 'c5'], 10.0),
        (['r3', 'c2'], 7.0),
        (['r7', 'c8'], 3.0),
        (['r9', 'c1'], 15.0),
        (['r5', 'c5'], 20.0),
    ]
    
    print("Non-zero elements:")
    for indices, expected in non_zero_tests:
        result = df_large[indices]
        status = "✓" if result == expected else "✗"
        print(f"  {status} df_large[{indices}] = {result}")
    
    # Test 5: Compare Mthd1 vs Mthd2 behavior
    print("\n--- Test 5: Comparing Mthd1 (dense) vs Mthd2 (sparse) ---")
    df_dense = Dataframe_Mthd1(sparse_data.copy(), index_labels, column_labels)
    
    print("Comparing same access on both implementations:")
    for indices, expected in test_cases[:3]:
        result_dense = df_dense[indices]
        result_sparse = df_sparse[indices]
        match = "✓" if result_dense == result_sparse == expected else "✗"
        print(f"  {match} {indices}: Dense={result_dense}, Sparse={result_sparse}, Expected={expected}")
    
    # Test 6: Edge cases
    print("\n--- Test 6: Edge cases ---")
    try:
        # Test with integer indices
        int_idx = [0, 1, 2]
        int_cols = [0, 1, 2]
        small_sparse = np.array([[0, 5, 0], [3, 0, 0], [0, 0, 7]], dtype=float)
        df_int = Dataframe_Mthd2(small_sparse, int_idx, int_cols)
        
        print(f"✓ Integer indices work: df_int[[0, 1]] = {df_int[[0, 1]]}")
        print(f"✓ Integer access: df_int[[1, 0]] = {df_int[[1, 0]]}")
    except Exception as e:
        print(f"✗ Error with integer indices: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)