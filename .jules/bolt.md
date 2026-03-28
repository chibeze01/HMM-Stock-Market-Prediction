## 2024-05-15 - Vectorized dictionary lookups vs Array indexing
**Learning:** Using `np.vectorize(dict.get)` to map values in a NumPy array is surprisingly slow, acting as an anti-pattern for performance since it loops in Python space.
**Action:** Always use direct array indexing (e.g., creating a mapping array where indices are keys and elements are values, then slicing it with the original array) when mapping contiguous integer keys. It's over 100x faster in this codebase.
