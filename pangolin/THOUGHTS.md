How should indexing work?

I think the best conceptual rule in terms of INTERFACE would be that:

1. Individual indices can be declared "orthogonal" or "vectorized"
2. All vectorized indices are broadcast against each other
3. The vectorized dimensions (if any) go at the beginning

The best notation I can think for this is "orthogonal by default"

```python
x = [5,7]
y = [2,3]
z = [6,7]

x = [5,7]; y = [2,3]; z = [6,7]

# 1. All Orthogonal (The Grid) -> 3D Result
B = A[x,y,z]            # B[i,j,k] = A[x[i], y[j], z[k]]

# 2. All Coupled (The Path) -> 1D Result
B = A[v[x], v[y], v[z]] # B[i]     = A[x[i], y[i], z[i]]

# 3. Mixed (Path + Grid) -> 2D Result
# x and z move together (dim 0, 2); y is independent (dim 1)
B = A[v[x], y, v[z]]    # B[i,j]   = A[x[i], y[j], z[i]]

# 4. Mixed (Grid + Path) -> 2D Result
# x and y move together (dim 0, 1); z is independent (dim 2)
B = A[v[x], v[y], z]    # B[i,j]   = A[x[i], y[i], z[j]]

# 5. Standard Slicing -> 3D Result
B = A[:, ::2, 3::]      # B[i,j,k] = A[i, 2*j, 3+k]

# 6. Coupled Slices (Main Diagonal) -> 1D Result
B = A[v[:], v[:], v[:]] # B[i]     = A[i, i, i]

# 7. Partial Diagonal (Trace) -> 2D Result
# Dims 0 and 2 are diagonal; Dim 1 is full/independent
B = A[v[:], :, v[:]]    # B[i,j]   = A[i, j, i]
```
