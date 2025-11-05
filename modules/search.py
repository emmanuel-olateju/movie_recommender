import numpy as np
from typing import Union

def binary_search(arr: Union[list, np.ndarray] , target: int) -> Union[int, float, None]:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return None  # not found