## 2024-05-15 - Vectorized dictionary lookups vs Array indexing
**Learning:** Using `np.vectorize(dict.get)` to map values in a NumPy array is surprisingly slow, acting as an anti-pattern for performance since it loops in Python space.
**Action:** Always use direct array indexing (e.g., creating a mapping array where indices are keys and elements are values, then slicing it with the original array) when mapping contiguous integer keys. It's over 100x faster in this codebase.
## 2026-03-29 - NumPy np.vectorize with dict lookups is an anti-pattern
**Learning:** Using `np.vectorize(dict.get)` to map values in a NumPy array executes a Python function call for every single element, which acts as a massive performance bottleneck. It defeats the entire purpose of using NumPy.
**Action:** When you need to map contiguous integer values (like state IDs) to other values, always use direct array indexing. Create a lookup array where the index corresponds to the key and the value is the mapped value, and use `lookup_array[input_array]` to perform the mapping in O(1) vectorized C code.
## 2024-03-31 - [Array Indexing Instead of Vectorize Dict Lookup]
**Learning:** `np.vectorize(dict.get)` is slow because it iterates over array elements in Python space. For contiguous integer lookups, creating a mapping array and indexing it directly with the source array (e.g., `mapping[hidden_states]`) is much faster and more idiomatic in NumPy.
**Action:** Replace `np.vectorize(dict.get)` with direct array indexing whenever mapping contiguous integer values like states to new values.
## 2024-04-01 - [Avoid np.vectorize with dict.get for contiguous integer mapping]
**Learning:** `np.vectorize` is essentially a Python-level `for` loop, meaning it iterates over items in Python space instead of compiling down to fast, vectorized C operations. When mapping contiguous integers (like Hidden States in an HMM) to values via a dictionary, doing `np.vectorize(dict.get)(array)` acts as a significant performance anti-pattern.
**Action:** Replace `np.vectorize(dict.get)` with direct array indexing. By initializing a mapping array where the array indices match the dictionary keys (e.g., `mapping_array = np.array([dict.get(i, 0) for i in range(max_key + 1)])`), you can instantly map all integers in C using `mapping_array[integers_array]`.
## 2024-05-24 - Vectorizing Gaussian Log Likelihood Computation
**Learning:** Replacing row-by-row iteration in Python list comprehensions with array-wide vectorized operations (e.g. `np.einsum('ni,ij,nj->n', diff, inv, diff)` for quadratic forms) entirely eliminates a performance bottleneck. It prevents redundant calculations, such as inverting covariance matrices and calculating determinants per-sample.
**Action:** When performing matrix operations on a list of samples against a common parameter (like a cluster mean/covariance), always pass the entire data array and use `np.einsum` to evaluate the expression over the 'N' dimension at once.
## 2024-06-10 - Avoiding Explicit Broadcasting in Pairwise Distances
**Learning:** Calculating pairwise distances using explicit broadcasting (e.g., `X[:, None, :] - means[None, :, :]`) creates a large, memory-intensive intermediate array of shape `(N, K, D)`. For large datasets, this can cause significant memory bottlenecks and slowdowns.
**Action:** To optimize pairwise distance calculations (like Euclidean distance for clustering), use the expanded squared distance formula with matrix multiplication: `-2 * np.dot(X, means.T) + np.sum(means**2, axis=1)`. This avoids explicit broadcasting and significantly reduces memory usage and execution time.

## 2024-05-25 - Avoid explicit broadcasting for pairwise Euclidean distance
**Learning:** Using explicit broadcasting to compute pairwise Euclidean distances (e.g. `X[:, None, :] - means[None, :, :]`) creates a huge intermediate `N x K x D` array. This causes massive memory allocations and drastically slows down execution.
**Action:** Expand the squared Euclidean distance formula (`||x||^2 + ||y||^2 - 2x^Ty`) to compute distances. Since we typically only need the argmin, we can ignore `||x||^2` and minimize `-2*np.dot(X, means.T) + np.sum(means**2, axis=1)`. This leverages highly optimized matrix multiplication and uses minimal memory.
## 2024-05-25 - Avoid explicit broadcasting for pairwise distances
**Learning:** Using explicit broadcasting like `X[:, None, :] - means[None, :, :]` to calculate pairwise distances in NumPy creates massive, memory-intensive intermediate 3D arrays, acting as a performance bottleneck.
**Action:** Use matrix multiplication instead. By expanding the squared distance formula `(x-y)^2 = x^2 - 2xy + y^2` and dropping the `x^2` term (since it's constant for argmin), you can compute pseudo-distances using `-2 * np.dot(X, means.T) + np.sum(means**2, axis=1)`. This is much faster and uses far less memory.
## 2025-05-18 - Avoid explicit loops for multi-dimensional event counts
**Learning:** Using a Python `for` loop with `zip` to iterate through an array and accumulate multi-dimensional transition counts (e.g., `trans[prev, nxt] += 1`) is an O(N) operation in Python space, acting as a performance bottleneck.
**Action:** Use `np.bincount` on flattened transition indices (e.g., `labels[:-1] * n_components + labels[1:]`) followed by `.reshape()`. This delegates the multi-dimensional accumulation entirely to optimized C code, yielding >15x speedups.
## 2024-05-26 - Vectorize HMM transition estimation
**Learning:** Estimating state transitions using a Python `for` loop and zip (e.g. `for prev, nxt in zip(...)`) creates significant overhead, especially for long sequences, acting as a performance bottleneck.
**Action:** Use `np.bincount` on flattened transition indices. For any sequence of integer states (0 to K-1), computing the 1D index array `labels[:-1] * K + labels[1:]` and counting frequencies with `np.bincount` converts an O(N) Python loop into an O(N) C-level operation, giving >50x speedups.
## 2024-05-30 - Vectorize trade simulation arrays using np.diff
**Learning:** Backtesting simulations using Python row-by-row iteration in a `for` loop on sequence data introduces unnecessary bottlenecking, especially on long histories. Explicit loop state tracking like `in_position` over the array is highly inefficient.
**Action:** Replace `for` loop iteration with `np.diff` on state/signal arrays. Using `np.where(np.diff(signals))` quickly finds entry and exit indices. Then use NumPy array assignments/slicing (e.g. `equity[entry:exit]`) coupled with vectorized calculations to optimize performance over the time dimension.
