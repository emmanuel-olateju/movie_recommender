import pytest
import numpy as np
import sys
from pathlib import Path

# Add the modules directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from sparse_matrix import MatrixEntry, DictVector, MatrixMthd_1 # type: ignore


class TestMatrixEntry:
    """Test the MatrixEntry class"""
    
    def test_matrix_entry_creation(self):
        entry = MatrixEntry(5.0, 2)
        assert entry.value == 5.0
        assert entry.col_idx == 2
    
    def test_matrix_entry_repr(self):
        entry = MatrixEntry(3.14, 1)
        assert "MatrixEntry" in repr(entry)
        assert "3.14" in repr(entry)


class TestDictVector:
    """Test the DictVector class"""
    
    def test_row_vector_creation_with_nonzero(self):
        vector = [1, 0, 3, 0, 5]
        row = DictVector(vector, 0)
        
        # Should only store non-zero entries
        assert len(row) == 3  # len() returns number of stored entries
        
        # CHANGED: .vector is now a dict {col_idx: value}, not list of MatrixEntry
        assert row.vector[0] == 1  # Direct value access
        assert row.vector[2] == 3
        assert row.vector[4] == 5
        
        # Column indices are the keys
        assert 0 in row.vector
        assert 2 in row.vector
        assert 4 in row.vector
        assert 1 not in row.vector  # Zero entries not stored
    
    def test_row_vector_indexing(self):
        """Test the __getitem__ indexing that returns values directly"""
        vector = [1, 0, 3, 0, 5]
        row = DictVector(vector, 0)
        
        # Access via indexing (returns values directly)
        assert row[0] == 1
        assert row[1] == 0  # Zero entry
        assert row[2] == 3
        assert row[3] == 0  # Zero entry
        assert row[4] == 5
    
    def test_row_vector_all_zeros(self):
        vector = [0, 0, 0, 0]
        row = DictVector(vector, 0)
        assert len(row) == 0  # No non-zero entries
        
        # All indices should return 0
        for i in range(4):
            assert row[i] == 0
    
    def test_row_vector_all_nonzero(self):
        vector = [1, 2, 3, 4]
        row = DictVector(vector, 1)
        assert len(row) == 4
        
        # CHANGED: Iterate over dict items
        for col_idx, value in row.vector.items():
            assert value == vector[col_idx]
    
    def test_row_vector_floats(self):
        vector = [1.5, 0.0, 3.7, 0.0]
        row = DictVector(vector, 0)
        assert len(row) == 2
        
        # CHANGED: Direct value access
        assert row.vector[0] == 1.5
        assert row.vector[2] == 3.7
        
        # Test indexing
        assert row[0] == 1.5
        assert row[1] == 0.0
        assert row[2] == 3.7
        assert row[3] == 0.0
    
    def test_row_vector_empty_raises_error(self):
        with pytest.raises(ValueError, match="DictVector cannot be empty"):
            DictVector([], 0)
    
    def test_row_vector_idx_tracking(self):
        vector = [5, 0, 7]
        row = DictVector(vector, 3)
        assert row.vector_idx == 3
        assert row.vidx == 3


class TestMatrixMthd_1:
    """Test the MatrixMthd_1 sparse matrix class"""
    
    def test_simple_matrix(self):
        # Create a simple sparse matrix
        np_matrix = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        
        sparse = MatrixMthd_1(np_matrix)
        
        # Check shape
        assert sparse.shape == (3, 3)
        
        # CHANGED: len() now returns number of stored rows (all 3 have values)
        assert len(sparse) == 3
        
        # CHANGED: Access rows and check their dict structure
        row0 = sparse[0]
        assert isinstance(row0, DictVector)
        assert len(row0) == 1  # One non-zero entry
        assert row0.vector[0] == 1
        
        row1 = sparse[1]
        assert isinstance(row1, DictVector)
        assert len(row1) == 1
        assert row1.vector[1] == 2
        
        row2 = sparse[2]
        assert isinstance(row2, DictVector)
        assert len(row2) == 1
        assert row2.vector[2] == 3
    
    def test_numpy_style_indexing(self):
        """Test numpy-style [i, j] indexing"""
        np_matrix = np.array([
            [1, 0, 3],
            [0, 2, 0],
            [4, 0, 5]
        ])
        
        sparse = MatrixMthd_1(np_matrix)
        
        # Test element access with tuple indexing
        assert sparse[0, 0] == 1
        assert sparse[0, 1] == 0  # Zero entry
        assert sparse[0, 2] == 3
        assert sparse[1, 0] == 0  # Zero entry
        assert sparse[1, 1] == 2
        assert sparse[1, 2] == 0  # Zero entry
        assert sparse[2, 0] == 4
        assert sparse[2, 1] == 0  # Zero entry
        assert sparse[2, 2] == 5
    
    def test_dense_matrix(self):
        # All non-zero entries
        np_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        sparse = MatrixMthd_1(np_matrix)
        
        assert sparse.shape == (2, 3)
        
        # Access rows
        row0 = sparse[0]
        row1 = sparse[1]
        assert len(row0) == 3
        assert len(row1) == 3
        
        # CHANGED: Verify all values using dict structure
        for col_idx, value in row0.vector.items():
            assert value == np_matrix[0, col_idx]
        
        for col_idx, value in row1.vector.items():
            assert value == np_matrix[1, col_idx]
        
        # Test numpy-style indexing
        for i in range(2):
            for j in range(3):
                assert sparse[i, j] == np_matrix[i, j]
    
    def test_all_zeros_matrix(self):
        np_matrix = np.array([
            [0, 0, 0],
            [0, 0, 0]
        ])
        
        sparse = MatrixMthd_1(np_matrix)
        
        assert sparse.shape == (2, 3)
        
        # CHANGED: len() returns stored rows (0 for all-zero matrix)
        assert len(sparse.rows) == 0  # No rows stored
        
        # CHANGED: Accessing empty rows returns scalar 0, not DictVector
        result0 = sparse[0]
        result1 = sparse[1]
        assert result0 == 0 or result0 == 0.0
        assert result1 == 0 or result1 == 0.0
        
        # All indexed values should return 0
        for i in range(2):
            for j in range(3):
                assert sparse[i, j] == 0
    
    def test_random_sparse_matrix(self):
        # Create a random sparse matrix
        np.random.seed(42)
        np_matrix = np.random.choice([0, 0, 0, 1, 2, 3], size=(5, 7))
        
        sparse = MatrixMthd_1(np_matrix)
        
        # Verify shape
        assert sparse.shape == (5, 7)
        
        # Verify each row
        for i in range(np_matrix.shape[0]):
            row = sparse[i]
            
            # Get non-zero indices from numpy
            nonzero_cols = np.nonzero(np_matrix[i])[0]
            
            # If row is all zeros, it returns 0, not DictVector
            if len(nonzero_cols) == 0:
                assert row == 0 or row == 0.0
                continue
            
            # CHANGED: Access dict directly
            assert isinstance(row, DictVector)
            assert len(row.vector) == len(nonzero_cols)
            
            # Verify each entry matches
            for col_idx, value in row.vector.items():
                assert np_matrix[i, col_idx] == value
        
        # Verify numpy-style indexing
        for i in range(np_matrix.shape[0]):
            for j in range(np_matrix.shape[1]):
                assert sparse[i, j] == np_matrix[i, j]
    
    def test_float_matrix(self):
        np_matrix = np.array([
            [1.5, 0.0, 2.7],
            [0.0, 0.0, 0.0],
            [3.14, 2.71, 0.0]
        ])
        
        sparse = MatrixMthd_1(np_matrix)
        
        # Row 0
        row0 = sparse[0]
        assert len(row0) == 2
        assert row0.vector[0] == 1.5
        assert row0.vector[2] == 2.7
        
        # Row 1 (all zeros) - returns scalar
        row1 = sparse[1]
        assert row1 == 0.0
        
        # Row 2
        row2 = sparse[2]
        assert len(row2) == 2
        assert row2.vector[0] == 3.14
        assert row2.vector[1] == 2.71
        
        # Test numpy-style indexing
        assert sparse[0, 0] == 1.5
        assert sparse[0, 1] == 0.0
        assert sparse[0, 2] == 2.7
        assert sparse[2, 0] == 3.14
        assert sparse[2, 1] == 2.71


class TestSparseMatrixEquivalence:
    """Test that sparse representation is equivalent to NumPy matrix"""
    
    def verify_sparse_matches_numpy(self, np_matrix, sparse_matrix):
        """Helper function to verify sparse matrix matches numpy matrix"""
        
        # Check shape
        assert sparse_matrix.shape == np_matrix.shape
        
        # CHANGED: Iterate over stored rows only
        for row_idx in range(np_matrix.shape[0]):
            sparse_row = sparse_matrix[row_idx]
            
            # Handle all-zero rows
            if not isinstance(sparse_row, DictVector):
                # Row is all zeros
                for col_idx in range(np_matrix.shape[1]):
                    assert np_matrix[row_idx, col_idx] == 0 or np_matrix[row_idx, col_idx] == 0.0
                continue
            
            # Get all non-zero entries from the sparse representation
            sparse_entries = sparse_row.vector  # Already a dict
            
            # Check each column in the numpy matrix
            for col_idx in range(np_matrix.shape[1]):
                np_value = np_matrix[row_idx, col_idx]
                
                if col_idx in sparse_entries:
                    # Non-zero entry exists in sparse matrix
                    assert np.isclose(sparse_entries[col_idx], np_value), \
                        f"Mismatch at ({row_idx}, {col_idx}): " \
                        f"sparse={sparse_entries[col_idx]}, numpy={np_value}"
                else:
                    # Entry doesn't exist in sparse matrix, should be zero in numpy
                    assert np_value == 0 or np_value == 0.0, \
                        f"Non-zero value {np_value} at ({row_idx}, {col_idx}) " \
                        f"missing from sparse representation"
                
                # Also verify numpy-style indexing
                assert sparse_matrix[row_idx, col_idx] == np_value or \
                       np.isclose(sparse_matrix[row_idx, col_idx], np_value), \
                       f"Indexing mismatch at ({row_idx}, {col_idx}): " \
                       f"sparse[{row_idx}, {col_idx}]={sparse_matrix[row_idx, col_idx]}, " \
                       f"numpy={np_value}"
    
    @pytest.mark.parametrize("shape,sparsity", [
        ((3, 3), 0.7),
        ((5, 5), 0.8),
        ((10, 10), 0.9),
        ((4, 6), 0.75),
    ])
    def test_various_sparse_matrices(self, shape, sparsity):
        """Test various sizes and sparsity levels"""
        np.random.seed(42)
        
        # Create a sparse matrix (sparsity = proportion of zeros)
        np_matrix = np.random.rand(*shape)
        np_matrix[np_matrix < sparsity] = 0
        
        sparse = MatrixMthd_1(np_matrix)
        self.verify_sparse_matches_numpy(np_matrix, sparse)
    
    def test_single_row_matrix(self):
        np_matrix = np.array([[1, 0, 3, 0, 5]])
        sparse = MatrixMthd_1(np_matrix)
        self.verify_sparse_matches_numpy(np_matrix, sparse)
    
    def test_single_column_matrix(self):
        np_matrix = np.array([[1], [0], [3], [0], [5]])
        sparse = MatrixMthd_1(np_matrix)
        self.verify_sparse_matches_numpy(np_matrix, sparse)
    
    def test_alternating_pattern(self):
        # Create a checkerboard-like pattern
        np_matrix = np.array([
            [1, 0, 1, 0],
            [0, 2, 0, 2],
            [3, 0, 3, 0],
            [0, 4, 0, 4]
        ])
        sparse = MatrixMthd_1(np_matrix)
        self.verify_sparse_matches_numpy(np_matrix, sparse)
    
    def test_negative_values(self):
        np_matrix = np.array([
            [-1, 0, 2],
            [0, -3, 0],
            [4, 0, -5]
        ])
        sparse = MatrixMthd_1(np_matrix)
        self.verify_sparse_matches_numpy(np_matrix, sparse)
    
    def test_very_small_nonzero_values(self):
        """Test that very small non-zero values are preserved"""
        np_matrix = np.array([
            [1e-10, 0, 1e-5],
            [0, 1e-8, 0]
        ])
        sparse = MatrixMthd_1(np_matrix)
        self.verify_sparse_matches_numpy(np_matrix, sparse)


class TestNumpyStyleIndexing:
    """Additional tests specifically for numpy-style indexing"""
    
    def test_single_index_returns_row(self):
        np_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        sparse = MatrixMthd_1(np_matrix)
        
        # Single index should return DictVector
        row0 = sparse[0]
        assert isinstance(row0, DictVector)
        assert len(row0.vector) == 3
    
    def test_tuple_single_index(self):
        np_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        sparse = MatrixMthd_1(np_matrix)
        
        # (i,) should return scalar (going through tuple indexing path)
        # Actually, this will work as expected if only one index provided
        value = sparse[1, 0]
        assert value == 4
    
    def test_out_of_bounds_behavior(self):
        """Test behavior when accessing indices that don't exist"""
        np_matrix = np.array([
            [1, 0, 3],
            [0, 2, 0]
        ])
        sparse = MatrixMthd_1(np_matrix)
        
        # Valid indices
        assert sparse[0, 0] == 1
        assert sparse[0, 1] == 0
        
        # Out of bounds should raise IndexError
        with pytest.raises(IndexError):
            _ = sparse[5, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])