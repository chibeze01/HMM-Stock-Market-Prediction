## 2024-05-24 - NumPy Vectorize vs Array Mapping
**Learning:** `np.vectorize(dict.get)` acts as an anti-pattern for performance as it iterates in Python space over the array values. For dictionaries where keys are contiguous integers, creating a NumPy array map where the array indices correspond to keys and using direct array mapping `mapping_arr[indices]` provides massive speedups.
**Action:** Always prefer direct array mapping `mapping_arr[keys]` instead of `np.vectorize` when mapping contiguous integer keys to values.
