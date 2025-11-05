import numpy as np
from typing import Union
from search import binary_search
from vector_methods import DictVector, ListVector

    
class MatrixMthd_2:

    def __init__(self, array: np.ndarray) -> None:
        self.shape = array.shape
        self.rows = []
        self.ridx = []
        self.item_type = array.dtype.type

        for j in range(array.shape[0]):
            if isinstance(array[j, :], list):
                vector = ListVector(array[j, :], j) # --> j here is the index of the row that might be created
            else:
                vector = ListVector(array[j, :].tolist(), j)

            if len(vector.vector)>0:
                self.rows.append(vector)
                self.ridx.append(j)

    def __getitem__(self, index: Union[int, tuple]) -> Union['ListVector', int, float]:
        if isinstance(index, int):
            if (index >= self.shape[0]) or (index < 0):
                raise IndexError(f"Row index {index} out of bounds for dimension 1 of shape: {self.shape}")
            
            idx = binary_search(self.ridx, index)
            if idx is None:
                if self.item_type is int:
                    return 0
                else:
                    return 0.0
            else:
                row = self.rows[idx]
                return row
        
        elif isinstance(index, tuple):
            if len(index) == 0:
                raise IndexError("Index tuple cannot be empty")
            
            if (index[0] >= self.shape[0]) or (index[0] < 0):
                raise IndexError(f"Row index {index} out of bounds for shape dimension 1 of shape: {self.shape}")
            
            idx = binary_search(self.ridx, index[0])
            if idx is None:
                if self.item_type is int:
                    return 0
                else:
                    return 0.0
            else:
                current = self.rows[idx]
                for dim_idx in range(1, len(index)):
                    idx = index[dim_idx]
                    
                    if isinstance(current, ListVector):
                        result = current[idx]

                        if (result is None) or (result == 0) or (result == 0.0):
                            if self.item_type is int:
                                return 0
                            else:
                                return 0.0
                            
                        current = result
                    else:
                        raise IndexError(f"Too many indices: matrix has {dim_idx} dimensions, but {len(index)} dimensions were indexed.")
                    
                return current
            
        else:
            raise TypeError(f"Invalid index type: {type(index)}, expected int or tuple")
    
class MatrixMthd_1:

    def __init__(self, array: np.ndarray) -> None:
        self.shape = array.shape
        self.rows = {}
        self.item_type =array.dtype.type

        for j in range(array.shape[0]):
            if isinstance(array[j, :], list):
                row = DictVector(array[j, :], j)
            else:
                row = DictVector(array[j, :].tolist(), j)
            # Only store rows that have at least one non-zero entry
            if len(row.vector) > 0:
                self.rows[j] = row

    def __getitem__(self, index: Union[int, tuple]) -> Union['DictVector', int, float]:
        if isinstance(index, int):
            if (index >= self.shape[0]) or (index < 0):
                raise IndexError(f"Row index {index} out of bounds for shape dimension 1 of shape: {self.shape}")
            
            row = self.rows.get(index)
            if row is None:
                if (self.item_type is int):
                    return 0
                else:
                    return 0.0
                
            return row
        
        elif isinstance(index, tuple):
            if len(index) == 0:
                raise IndexError("Index tuple cannot be empty")
            
            if (index[0] >= self.shape[0]) or (index[0] < 0):
                raise IndexError(f"Row index {index} out of bounds for shape dimension 1 of shape: {self.shape}")

            current = self.rows.get(index[0])
            if current is None:
                if self.item_type is int:
                    return 0
                else:
                    return 0.0

            for dim_idx in range(1, len(index)):
                col_idx = index[dim_idx]
                if isinstance(current, DictVector):
                    result = current[col_idx]
                
                    if (result is None) or (result == 0) or (result == 0.0):
                        if self.item_type is int:
                            return 0
                        else:
                            return 0.0
                
                    current = result
                else:
                    print(f"Type raising error: {type(current)}")
                    raise IndexError(f"Too many indices: matrix has {dim_idx} dimensions, but {len(index)} dimensions were indexed.")

            return current
            
        else:
            raise TypeError(f"Invalid index type: {type(index)}, expected int or tuple")
    
    def __len__(self) -> int:
        return len(self.rows)

    def to_dense(self) -> np.ndarray:
        result = np.zeros(self.shape, dtype=self.item_type)

        for row_idx, row in self.rows.items():
            for col_idx, value in row.vector.items():
                if isinstance(value, DictVector):
                    result[row_idx, col_idx] = value    # --> Need to do recursively to allow for multi-dimensional sparse matrices
                else:
                    result[row_idx, col_idx] = value

        return result
    
    def __repr__(self):
        return f"MatrixMthd_1(shape={self.shape}, stored_rows={self.rows}, density={len(self.rows)/self.shape[0]:.2%}, matrix: {self.to_dense()})"
    





if __name__ == "__main__":
    print("=" * 60)
    print("Testing 2D Sparse Matrix")
    print("=" * 60)
    
    # Test 2D array
    array_2d = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 0]
    ])
    
    matrix = MatrixMthd_1(array_2d)
    print(f"\nOriginal array:\n{array_2d}")
    print(f"\nSparse representation: {matrix}")
    
    # Test accessing elements
    print(f"\nmatrix[1, 0] = {matrix[1, 0]} (stored)")
    print(f"matrix[0, 0] = {matrix[0, 0]} (zero, not stored)")
    print(f"matrix[3, 2] = {matrix[3, 2]} (stored)")
    print(f"matrix[2, 2] = {matrix[2, 2]} (zero, not stored)")
    
    # Test accessing entire row
    print(f"\nmatrix[1] = {matrix[1]} (non-empty row)")
    print(f"matrix[0] = {matrix[0]} (empty row)")
    
    print("\n" + "=" * 60)
    print("Testing 3D Sparse Matrix")
    print("=" * 60)
    
    # Test 3D array
    array_3d = np.array([
        [[1, 0], [0, 0]],
        [[0, 0], [0, 5]],
        [[0, 0], [0, 0]]
    ])
    
    matrix_3d = MatrixMthd_1(array_3d)
    print(f"\nOriginal 3D array shape: {array_3d.shape}")
    print(f"Sparse representation: {matrix_3d}")
    
    print(f"\nmatrix_3d[0, 0, 0] = {matrix_3d[0, 0, 0]} (stored)")
    print(f"matrix_3d[1, 1, 1] = {matrix_3d[1, 1, 1]} (stored)")
    print(f"matrix_3d[0, 1, 0] = {matrix_3d[0, 1, 0]} (zero, not stored)")
    print(f"matrix_3d[2, 0, 0] = {matrix_3d[2, 0, 0]} (zero, not stored)")
    
    print("\n" + "=" * 60)
    print("Memory Efficiency Test")
    print("=" * 60)
    
    # Large sparse matrix
    large_sparse = np.zeros((1000, 1000))
    large_sparse[10, 50] = 99
    large_sparse[500, 700] = 88
    large_sparse[999, 999] = 77
    
    matrix_large = MatrixMthd_1(large_sparse)
    print(f"\nMatrix shape: {matrix_large.shape}")
    print(f"Total possible rows: {matrix_large.shape[0]}")
    print(f"Rows actually stored: {len(matrix_large.rows)}")
    print(f"Memory efficiency: Only {len(matrix_large.rows)/matrix_large.shape[0]:.2%} of rows stored!")
    
    print(f"\nAccessing stored value: matrix[10, 50] = {matrix_large[10, 50]}")
    print(f"Accessing zero value: matrix[0, 0] = {matrix_large[0, 0]}")