---
layout: post
title: "PyTorch Cheatsheet"
date: 2025-06-15
tags: pytorch, deep-learning
citation: true
toc:
  sidebar: left
---
This assumes prior experience with PyTorch or related frameworks. The content is presented in a dense manner so that it can serve as a quick reference. Readers are encouraged to run the code themselves to get a better understanding. 

AI was heavily involved in some sections. However, AI still doesn't have a strong understanding of niche PyTorch behaviors, so I have had to make many corrections. I welcome any feedback about inaccuracies or suggestions for general improvements.

# Tensor Creation

### Tensor Creation Methods

```python
# Basic tensor creation
torch.zeros(2, 3)  # all zeros
torch.ones(2, 3)   # all ones
torch.empty(2, 3)  # uninitialized memory
torch.full((2, 3), 7)  # fill with value

# Random tensors
torch.randn(2, 3)  # normal distribution
torch.rand(2, 3)   # uniform [0, 1)
torch.randint(0, 10, (2, 3))  # random integers
torch.randperm(10)  # random permutation

# From existing data
torch.tensor([1, 2, 3])  # from list
torch.from_numpy(numpy_array)  # from numpy
torch.as_tensor(data)  # shares memory if possible

# Match existing tensor
x = torch.randn(2, 3)
torch.zeros_like(x)  # same shape/dtype
torch.ones_like(x)
torch.randn_like(x)

# Sequences
torch.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)   # evenly spaced
torch.logspace(0, 2, 3)   # log spaced

# Special matrices
torch.eye(3)  # identity matrix
torch.diag(torch.tensor([1, 2, 3]))  # diagonal matrix
```

### Tensor Data Types

```python
# Common data types
torch.float32  # default float
torch.float64  # double precision
torch.float16  # half precision for memory efficiency
torch.int64    # default integer
torch.int32    # 32-bit integer
torch.bool     # boolean values
torch.complex64  # complex numbers

# Type conversion
x = torch.randn(2, 3)
x.float()   # to float32
x.double()  # to float64
x.int()     # to int32
x.long()    # to int64
x.bool()    # to boolean

# Creating with specific dtype
torch.zeros(2, 3, dtype=torch.float16)
torch.tensor([1, 2, 3], dtype=torch.long)

# Check dtype
x.dtype
x.is_floating_point()
```

### Random Number Generation and Seeding

```python
# Set seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42) # For CUDA operations
torch.cuda.manual_seed_all(42) # For all CUDA devices

# Generator objects for more control
g = torch.Generator()
g.manual_seed(42)
torch.randn(2, 3, generator=g) # random seed set for this single operation

# Random number distributions
torch.normal(mean=0, std=1, size=(2, 3)) # Normal distribution
torch.exponential(torch.ones(2, 3)) # Exponential distribution
torch.poisson(torch.ones(2, 3)) # Poisson distribution
```

### Device Management

```python
# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check for MPS (Apple Silicon)
if torch.backends.mps.is_available():
   device = torch.device('mps')

# Create tensors on specific devices
x_cpu = torch.randn(2, 3)  # default: CPU
x_gpu = torch.randn(2, 3, device='cuda')  # directly on GPU
x_gpu = torch.randn(2, 3).cuda()  # move to GPU after creation

# Move between devices
x_cpu = x_gpu.cpu()  # GPU to CPU
x_gpu = x_cpu.to(device)  # CPU to device
x_gpu = x_cpu.to('cuda:0')  # specific GPU

# Check tensor device
x.device
x.is_cuda

# Multiple GPU handling
if torch.cuda.device_count() > 1:
   x_gpu1 = x.to('cuda:1')  # move to second GPU
```

---

# Tensor Fundamentals

## Broadcasting Semantics

Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions.

**Basic Rules:**

- Dimensions are aligned from the right (last dimension)
- Each dimension pair must either: be equal, or one is size 1, or one is missing
- Output has the maximum size in each dimension

The shorter dimension is padded with singleton dimensions (size 1) from the left. Where one tensor has a singleton dimension, and the other tensor has a dimension size greater than 1, the first tensor is repeated along that dimension to match.

**Example:**

```
A: [    5, 3, 1]
B: [8, 1, 3, 4]
   ↓  ↓  ↓  ↓
C: [8, 5, 3, 4]
```

Broadcasting also works with scalars, in that a scalar can be repeated to match the dimensions of a tensor.

Note that this explanation is how we can conceptually think of broadcasting operations but it may be implemented in a more efficient way in PyTorch.

**Implementation:**

```python
# Tensor-Scalar Broadcasting
x = torch.randn(3, 4)
scalar = 2.0
result = x + scalar  # scalar broadcasts to shape (3, 4)

# Tensor-Tensor Broadcasting
A = torch.randn(3, 1)  # shape: (3, 1)
B = torch.randn(1, 4)  # shape: (1, 4)
C = A + B  # result shape: (3, 4)

# Tensor-Vector Broadcasting
matrix = torch.randn(3, 4)  # shape: (3, 4)
vector = torch.randn(4)     # shape: (4,)
result = matrix + vector    # vector broadcasts across rows

# Explicit broadcasting
D = A.expand(3, 4) + B.expand(3, 4)
```

### Tensor Expansion

PyTorch provides built-in operations for repeating tensors, which allows for more explicit implementation of broadcasting.

```python
# Implicit broadcasting
A = torch.randn(3, 1)
B = torch.randn(1, 4)
C = A + B # [3, 4]

# Explicit broadcasting
D = A.expand(3, 4) + B.expand(3, 4)
```

There are different ways to expand / repeat a tensor:

```python
# Different ways to expand a (1,4) tensor to (3,4)
x = torch.tensor([[1, 2, 3, 4]])  # shape: (1, 4), also works with (4,)

# repeat() - creates new memory (independent copy)
repeated = x.repeat(3, 1)  # shape: (3, 4)
repeated[0, 0] = 999  # only repeated changes, x unchanged

# expand() - memory efficient view (shares data with original)
# This applies logic similar to broadcasting to determine which dimension
# to expand on. It can only exapnd singleton or non-existent dimensions.
expanded = x.expand(3, 4)  # shape: (3, 4)
expanded[0, 0] = 888 # both x and expanded change due to shared memory
# Each repeated row shares the same memory

# broadcast_to() - function equivalent of torch.expand
x = torch.tensor([[1, 2, 3, 4]])
broadcasted = torch.broadcast_to(x, (3, 4))  # shape: (3, 4)

# tile() - repeats data (new memory)
tiled = torch.tile(x, (2, 3))  # shape: (2, 12)
# [1,2,3,4] repeated 3 row-wise and 2 times column wise
```

## Indexing and Slicing

PyTorch does not have good documentation on indexing, but it is mostly similar to NumPy: [https://numpy.org/devdocs//user/basics.indexing.html](https://numpy.org/devdocs//user/basics.indexing.html)

[https://www.tensorflow.org/guide/basics](https://www.tensorflow.org/guide/basics)

### **Basic Indexing**

- Slicing with a range keeps the dimension
- Scalar index removes the dimension

```python
tensor = torch.tensor([[1, 2], [3, 4]])  # (2, 2)

# Different indexing results
tensor[1, 2]               # scalar
tensor[1, :]               # (2,)
tensor[1:2, :]             # (1, 2)
tensor[1, :].unsqueeze(0)  # (1, 2)
tensor[1, :].unsqueeze(-1) # (2, 1)
tensor[1, :][None, :]      # (1, 2)

tensor[0:3:2, :]           # step slicing
tensor[-1, -2:]            # last row, last two columns
```

**Views**

Basic indexing creates views. This means the output uses the same memory. We can see that modifying a view will also modify the original tensor

```python
tensor = torch.zeros((2,2))
view = tensor[:,1]
view[:] = 1
tensor  # [[0., 1.], [0., 1.]]

# Same data, different metadata
tensor.data_ptr() == view.data_ptr()  # True - shared memory

tensor.stride()  # (2, 1) - row skip 2, col skip 1  
view.stride()    # (2,)   - skip 2 for next element

tensor.storage_offset()  # 0 - starts at storage[0]
view.storage_offset()    # 1 - starts at storage[1]
```

### **Advanced Indexing**

These methods generate **copies** of the data instead of views. In some applications, this can cause additional memory usage. 

**Tensor Indexing**

Tensor indexing is when you use a tensor at a specific dimension. This can be thought of as a generalization of basic indexing. For example you can express [1,2,3,5] with basic indexing, but not [1,5,6,9]. It is not possible to generate a view when the strides vary like this. 

Advanced indexing with multiple tensors broadcasts the index arrays to create an implicit coordinate grid. The broadcast shape becomes your output shape, where each position maps to a specific element selection from the original tensor. Think of it as a "virtual mask" - PyTorch computes only the needed (row, col) coordinates without materializing a full boolean mask, efficiently selecting elements based on the broadcasted index pair.

If we use a single index tensor, the shape of the corresponding dimension is updated to the size of the index tensor.

```python
tensor = torch.randn(3, 3)

# Single tensor indexing
row_indices = torch.tensor([0, 2])
tensor[row_indices]  # shape: (2, 3)

# Higher dimensional indexing.
# We can use a higher dimension tensor at any position.
# This will add dimensions to the output
indices_2d = torch.tensor([[0, 1], [2, 0]])  # shape: (2, 2)
tensor[indices_2d, :]  # shape: (2, 2, 3)
tensor[:, indices_2d]  # shape: (3, 2, 2)

# Higher dimensional indices just specify the shape 
# we want the output to be at that dimension
# Equivalent to flatten -> index -> reshape
tensor[indices_2d.flatten(), :].reshape(2, 2, 3)  # Same as tensor[indices_2d, :]

# List Indexing
# Integer lists and tensors are interchangeable
tensor[[0, 2]]  # shape: (2, 3)
```

 If we use multiple index tensors, the shapes of each index tensor are broadcast together. The resulting shape is added to the position of the indices are contiguous, otherwise they are added the beginning.

```python
tensor = torch.randn(4, 3)
# Multiple tensor indexing - can repeat indices
row_idx = torch.tensor([0, 0, 2])  # shape: (3,)
col_idx = torch.tensor([1, 2, 0])  # shape: (3,)
tensor[row_idx, col_idx]  # shape: (3,)

# Different shapes broadcast together
rows = torch.tensor([[0], [2]])  # shape: (2, 1)
cols = torch.tensor([0, 2])      # shape: (2,)
tensor[rows, cols]  # broadcasts to (2, 2)

# Non-contiguous Dimensions
tensor = torch.randn(4, 4, 5, 6)
# 4D tensor with non-adjacent indexed dimensions
dim1_idx = torch.tensor([0, 2])  # shape: (2,)
dim3_idx = torch.tensor([1, 4])  # shape: (2,)

# Same shape indices - pairs elements
tensor[:, dim1_idx, :, dim3_idx]  # shape: (4, 2, 5)

# Different shapes - broadcast
dim1_broad = torch.tensor([0, 2]).unsqueeze(1)  # shape: (2, 1)
dim3_broad = torch.tensor([1, 4])               # shape: (2,)
tensor[:, dim1_broad, :, dim3_broad]  # shape: (4, 2, 2, 5)

# Mixed Basic + Advanced Indexing
row_indices = torch.tensor([0, 2])
tensor[row_indices, :]  # shape: (2, 4, 5, 6)
tensor[:, row_indices]  # shape: (4, 2, 5, 6)

```

**Mask Indexing**

Mask indices create an additional dimension that is of the size of the number of selected elements true values in the mask. This dimension is added at the position of the mask. 

There is no broadcasting so the size of the mask must match at the position it is applied.

```python
tensor = torch.randn(3, 4)

# Boolean condition indexing - always returns 1D
mask = tensor > 0
tensor[mask]  # shape: (N,) where N = number of True values

# Boolean tensor indexing
mask = torch.tensor([True, False, True])
tensor[mask]  # shape: (2, 4) - selects rows 0 and 2

# 2D boolean mask - flattens result
mask = torch.tensor([[True, False, True, False],
                      [False, True, False, True], 
                      [True, True, False, False]])
tensor[mask]  # shape: (5,) - flattened selection

# Mixed boolean + basic indexing
tensor[mask, :]     # shape: (2, 4) - same as tensor[mask2]
tensor[:, mask[:4]] # shape: (3, 2) - boolean on columns

# Boolean mask on higher dimensions
data = torch.randn(2, 3, 4)
mask_3d = data > 0.5
data[mask_3d]  # shape: (M,) - flattened regardless of input shape

# Boolean mask with specific dimensions
mask_2d = torch.tensor([[True, False], [False, True], [True, False]])
data_2d = torch.randn(3, 2, 5)
data_2d[mask_2d]  # shape: (3, 5) - selects elements where mask is True
```

**Mask vs Tensor Indexing**

We can convert between mask and tensor indexing. When converting a mask to index tensors, we want the length of the index tensor to match the number of True values in the mask. When going the other direction, this may require repeating index values to reconstruct the mask. 

```python
tensor = torch.randn(3, 4)

# Boolean mask to tensor indices
mask = torch.tensor([[True, False, True, False], 
                    [False, True, False, True], 
                    [True, True, False, False]])

row_idx, col_idx = torch.where(mask)
tensor[mask]                    # boolean indexing: shape (5,)
tensor[row_idx, col_idx]        # tensor indexing: shape (5,)

# Tensor indices to boolean mask
indices_row = torch.tensor([0, 1, 2, 0, 1])
indices_col = torch.tensor([0, 1, 3, 2, 1])

mask_reconstructed = torch.zeros(3, 4, dtype=torch.bool)
mask_reconstructed[indices_row, indices_col] = True

tensor[indices_row, indices_col]    # tensor indexing: shape (5,)
tensor[mask_reconstructed]          # boolean indexing: shape (4,) - one less due to duplicated indices

# These two operations are not identical since the boolean mask loses the order 
# of the indices used in the tensor indexing. We also remove duplicates when using bool
```

**Gather / Scatter**

The gather operation allows getting values based on indices. This best suited to get individual values out of a tensor rather than vectors/tensors. This requires an index tensor of the same number of dimensions and a dim argument to specify which dimension to get the values from. The shape of the output is the same as the shape of the index.

Scatter is the inverse in that you take indices and values and write the values to a target tensor at the specified indices. The shape of the indices and source tensor are the same.

In both gather and scatter, the dim argument specifies the axis along which the actual indexing or writing happens. For all dimensions other than dim, the structure of the index tensor determines how values are aligned and positioned. In gather, values are selected from the input tensor along dim using the indices in the index tensor, while the rest of the dimensions follow the layout of index. In scatter, values from the src tensor are written into the input tensor at positions specified by index along dim, with all other dimensions used to align corresponding values from src and index.

```python
# Gather example
src = torch.tensor([[1, 2, 3], [4, 5, 6]])
index = torch.tensor([[0, 1, 1], [0, 0, 1]])
out = torch.gather(src, 1, index)  # dim=1 gathers along rows
# out: tensor([[1, 2, 2], [4, 4, 5]])
out = torch.gather(src, 0, index) # dim=0 gathers along cols
# out_dim0: tensor([[1, 5, 3], [4, 2, 6]])

# Scatter example 
dest = torch.zeros(2, 3)
index = torch.tensor([[0, 1, 1], [0, 0, 1]])
src = torch.tensor([[1, 2, 3], [4, 5, 6]])
dest.scatter_(1, index, src)  # dim=1 scatters along rows
# dest: tensor([[1, 2, 3], [4, 5, 6]])
```

To better illustrate this, we can consider how this would work if implemented using for loops:

```python
def gather_manual(src, dim, index):
    """Manual implementation of torch.gather using for loops"""
    output = torch.zeros_like(index, dtype=src.dtype)
    
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            if dim == 0:
                # Gather along rows: fix column j, vary row
                output[i, j] = src[index[i, j], j]
            elif dim == 1:
                # Gather along columns: fix row i, vary column  
                output[i, j] = src[i, index[i, j]]
    
    return output

def scatter_manual(dest, dim, index, src):
    """Manual implementation of torch.scatter using for loops"""
    result = dest.clone()  # Don't modify original
    
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            if dim == 0:
                # Scatter along rows: fix column j, vary row
                result[index[i, j], j] = src[i, j]
            elif dim == 1:
                # Scatter along columns: fix row i, vary column
                result[i, index[i, j]] = src[i, j]
    
    return result
    
# example of gather returning the original tensor
input = torch.tensor([[10, 20, 30],
                      [40, 50, 60]])  # shape (2, 3)

# Create an identity index along dim=1
index = torch.arange(input.size(1)).expand_as(input)
# index = [[0, 1, 2],
#          [0, 1, 2]]

output = torch.gather(input, dim=1, index=index)
```

We can also implement similar operations with eye.

```python
# Scatter-add using identity matrix: sum values to target positions
values = torch.tensor([10., 20., 30.])  # values to scatter
link = torch.tensor([0, 2, 0])          # target positions [0,2,0] 
j = 3                                    # output size

# eye(j)[link] creates selection matrix with one-hot rows
# [[1,0,0],  <- row 0 (link[0]=0)
#  [0,0,1],  <- row 2 (link[1]=2)  
#  [1,0,0]]  <- row 0 (link[2]=0)

result = values @ torch.eye(j)[link]     # [40, 0, 20] - automatic sum at pos 0

# Related identity matrix tricks:
source = torch.tensor([100., 200., 300.])
indices = torch.tensor([0, 2, 0])

gathered = torch.eye(len(source))[indices] @ source        # gather: [100, 300, 100]
one_hot = torch.eye(3)[torch.tensor([1, 0, 2])]            # one-hot encoding
permuted = torch.eye(3)[torch.tensor([2, 0, 1])] @ source  # reorder: [300, 100, 200]

# Key: eye(n)[indices] transforms indexing → matrix multiplication
```

**Scatter Reduction Operations**

PyTorch's scatter operations support various reduction modes when multiple source values are scattered to the same target location.

```python
dest = torch.zeros(3, 4)
index = torch.tensor([[0, 1, 1, 2],
                      [1, 0, 1, 2], 
                      [2, 2, 0, 0]])  # Multiple indices point to same positions
src = torch.tensor([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]], dtype=torch.float)

# Regular scatter (overwrites, only keeps last value)
dest_regular = dest.clone()
dest_regular.scatter_(1, index, src)

# Scatter with specific operations
dest_add = dest.clone()
dest_add.scatter_add_(1, index, src)  # Equivalent to scatter_reduce with 'sum'

dest_sum = dest.clone() 
dest_sum.scatter_reduce_(1, index, src, reduce='sum')  # Same as scatter_add_

dest_mean = dest.clone()
dest_mean.scatter_reduce_(1, index, src, reduce='mean')

dest_max = dest.clone()
dest_max.scatter_reduce_(1, index, src, reduce='amax')
```

The reduction operation is performed in the order of the source elements. This means that for operations like max/min, the last value written to an index will be compared with the current value at that position.

### **Ellipsis Indexing**

Shorthand for “all remaining dimensions”. This can be used for both basic and advanced indexing.

```python
x = torch.randn(2, 3, 4, 5, 6)

# Basic indexing
first_last = x[0, ..., 1]   # equivalent to x[0, :, :, :, 1]
edge_dims = x[0, ..., 0]    # shape: (3, 4, 5)

# Advanced indexing with ellipsis
indices = torch.tensor([0, 2])
mask = torch.tensor([True, False, True, False, True])

# Ellipsis with integer tensor (advanced)
result1 = x[indices, ...]              # shape: (2, 3, 4, 5, 6)
result2 = x[..., indices]              # shape: (2, 3, 4, 5, 2) 
result3 = x[0, indices, ..., 1]        # shape: (2, 4, 5)

# Ellipsis with boolean mask (advanced)
result4 = x[..., mask]                 # shape: (2, 3, 4, 3) - 3 True values
result5 = x[0, ..., mask]              # shape: (3, 4, 3)

# Mixed advanced indexing
result6 = x[indices, ..., mask]        # shape: (2, 3, 4, 3)
result7 = x[0, indices, ..., mask]     # shape: (2, 4, 3)
```

## **Assignment**

In tensor assignment, the source automatically broadcasts to match the target's shape.

```
target = tensor[index]  
tensor[index] = source  <- source broadcasts to target's shape
```

We can understand assignment through each type of indexing through this framework. Boolean mask indexing is a bit more complicated because there is a new dimension that is formed by the number of values for which the mask is true. The source tensor must also have this dimension. The values from the target are written to the source in row major order. 

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float)  # (2, 3)
                  
# () indicates the shape of a scalar tensor

# Basic indexing
x[0, 1] = 10                 # target: () -> source: scalar -> ()
x[0, :] = 100                # target: (3,) -> source: scalar -> (3,)
x[:, :] = torch.randn((2,1)) # target: (3,) -> source: (2,1) -> (2,3)
x[:, :] = torch.randn((2,))  # error: target: (3,) -> source: (2,) can't broadcast

# Advanced indexing
idx = torch.tensor([0, 1])
x[idx, :] = torch.randn((2,3))  # target: (2, 3) -> source: (2, 3) -> (2, 3)
x[idx, :] = torch.randn((1,3))  # target: (2, 3) -> source: (3,) -> (2, 3)

# Boolean mask indexing
tensor = torch.randn((4,6,8))
get_mask = lambda mask_shape: torch.bernoulli(torch.rand(mask_shape)).bool()

# mask last two dimensions and broadcast across the first two
mask = get_mask((6, 8))
num_valid_values = mask.sum()
# target: (4, num_valid_values) -> source: (1, num_valid_values) -> (4, num_valid_values)
tensor[:, mask] = torch.zeros((1, num_valid_values), dtype=torch.float) 
# target: (4, num_valid_values) -> source: (4, num_valid_values) -> (4, num_valid_values)
tensor[:, mask] = torch.zeros((4, num_valid_values), dtype=torch.float) 
```

**dtype consistency**

PyTorch requires that the dtype of the source and target to be the same with advanced indexing. However, it is more lenient and casts dtypes when possible.

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # int64 tensor

# Basic indexing - dtype conversion allowed
x[:, 0:1] = torch.randn((2, 1))  # float auto-converts to int

# Advanced indexing - strict dtype matching required
idx = torch.tensor([0, 1])
x[idx, :] = torch.randn((2, 3)) # RuntimeError: Index put requires the source and destination dtypes match
x[idx, :] = torch.randn((2, 3)).int()
```

## Strategies

These are different common types of advanced operations that serve as case studies to how advanced indexing can work.

**Meshgrid Indexing**

This strategy involves using arange to repeat an index along some dimensions. This allows us to manually specify an index tensor at a specific position that varies depending on the position in the other indices.

```python
B, C, H, W = 2, 4, 3, 3
x = torch.arange(B * C * H * W).reshape(B, C, H, W)

# Say we want to gather 1 channel per pixel (like an argmax over channels)
channel_idx = torch.randint(0, C, (B, H, W))  # shape: (B, H, W)

# Manually build index tensors
b = torch.arange(B).view(B, 1, 1).expand(-1, H, W)
h = torch.arange(H).view(1, H, 1).expand(B, H, W)
w = torch.arange(W).view(1, 1, W).expand(B, H, W)
# use torch.meshgrid
b, h, w = torch.meshgrid(
    torch.arange(B),
    torch.arange(H),
    torch.arange(W),
    indexing='ij'  # important: ensures correct order
)

result = x[b, channel_idx, h, w]

# alternative torch.gather
channel_idx_expanded = channel_idx.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
result = torch.gather(x, dim=1, index=channel_idx_expanded)  # (B, 1, H, W)
result = result.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
```

**Top K Gathering**

```python
B, L, num_experts, d_model = 2, 3, 5, 4
expert_outputs = torch.randn(B, L, num_experts, d_model)  # (2, 3, 5, 4)
routing_scores = torch.randn(B, L, num_experts)           # (2, 3, 5)

# Select top-k experts per token
k = 2
_, expert_indices = torch.topk(routing_scores, k=k, dim=2)  # (2, 3, 2)

# Method 1: Use repeat instead of unsqueeze + expand (cleaner)
expanded_index = expert_indices.unsqueeze(-1).repeat(1, 1, 1, d_model)
result1 = torch.gather(expert_outputs, dim=2, 
                      index=expanded_index)

# Method 2: Use advanced indexing (most readable)
b_idx = torch.arange(B)[:, None, None]  # (B, 1, 1)
l_idx = torch.arange(L)[None, :, None]  # (1, L, 1)
result2 = expert_outputs[b_idx, l_idx, expert_indices]  # (2, 3, 2, 4)

# Method 3: Use meshgrid
# First create coordinate tensors
coords = torch.meshgrid(torch.arange(B), torch.arange(L), indexing='ij')
result3 = expert_outputs[coords[0][:, :, None], coords[1][:, :, None], expert_indices]

# Top K with Mask
B, N = 3, 4
scores = torch.randn(B, N)  # (batch_size, num_items) 
mask = torch.randint(0, 2, (B, N)).bool()  # Which items are "active"
k = 3

scores[~mask] = float('-inf')
topk_values, topk_indices = torch.topk(scores, k=k, dim=-1)
# set indices of invalid values to -1
# this handles the case when there are less than k active indices
topk_indices[topk_values==float('-inf')] = -1
```

**Selecting Diagonal**

```python
# Assigning the diagonal
B, W = 2, 4
tensor = torch.zeros((B,W,W))

# Method 1: Integer array indexing
tensor[torch.arange(B)[:, None, None], 
       torch.arange(W)[None, :], 
       torch.arange(W)[None, :]] = torch.arange(W, dtype=torch.float)

# Method 2: Boolean mask indexing
mask = torch.eye(W, dtype=torch.bool)
tensor[:, mask] = torch.arange(W, dtype=torch.float)

# Method 3: torch.diagonal() with copy assignment (in-place)
torch.diagonal(tensor, dim1=1, dim2=2).copy_(torch.arange(W, dtype=torch.float))

# Method 4: Scatter operation (in-place)
diag_indices = torch.arange(W).unsqueeze(0).unsqueeze(-1).expand(B, W, 1)  # (B, W, 1)
diag_values = torch.arange(W, dtype=torch.float).unsqueeze(0).unsqueeze(-1).expand(B, W, 1)  # (B, W, 1)
tensor.scatter_(2, diag_indices, diag_values)

# Method 5: Manual coordinate creation (in-place)
b_idx, diag_idx = torch.meshgrid(torch.arange(B), torch.arange(W), indexing='ij')
tensor[b_idx, diag_idx, diag_idx] = torch.arange(W, dtype=torch.float)
```

**Selecting Multiple Diagonals**

```python
B, W = 2, 6
tensor = torch.zeros((B,W,W))

# Select diagonals based on parameterized offsets
diagonal_offsets = [-1, 0, 2] # 1 below main diagonal, main, 2 above

# Method 1: Meshgrid
i, j = torch.meshgrid(torch.arange(W), torch.arange(W), indexing='ij')
diag_mask = torch.zeros_like(i, dtype=torch.bool)
for offset in diagonal_offsets:
    diag_mask |= (j - i == offset)
tensor[:, diag_mask] = torch.randn((B, diag_mask.sum()))

# Method 2: Broadcasting
# construct diag mask manually and broadcast
i = torch.arange(W).unsqueeze(1)  # (W, 1)
j = torch.arange(W).unsqueeze(0)  # (1, W)
# same usage of i and j

# Method 3: Triangular matrices
mask = torch.zeros((W, W), dtype=torch.bool)
for offset in diagonal_offsets:
    if offset >= 0:
        # Isolate specific upper diagonal
        temp_mask = torch.triu(torch.ones((W, W), dtype=torch.bool), diagonal=offset)
        temp_mask &= ~torch.triu(torch.ones((W, W), dtype=torch.bool), diagonal=offset+1)
    else:
        # Isolate specific lower diagonal
        temp_mask = torch.tril(torch.ones((W, W), dtype=torch.bool), diagonal=offset)
        temp_mask &= ~torch.tril(torch.ones((W, W), dtype=torch.bool), diagonal=offset-1)
    mask |= temp_mask
tensor[:, mask] = torch.randn((B, mask.sum()))

# Method 4: Use torch.where for assignment
row_idx, col_idx = torch.where(mask)
tensor[:, row_idx, col_idx] = torch.randn((B, len(row_idx)))

# Method 5: Vectorized masking (check all offsets at once)
differences = j - i
offset_tensor = torch.tensor(diagonal_offsets)
matches_offset = differences.unsqueeze(-1) == offset_tensor # bool tensor of size (W, W, num_offsets)
mask = matches_offset.any(-1) # reduce num_offsets dimension
tensor[:, mask] = torch.randn((B, mask.sum()))

# Method 6: isin instead of any
mask = torch.isin(differences, torch.tensor(diagonal_offsets))
tensor[:, mask] = torch.randn((B, mask.sum()))
```

**Sliding Window**

```python
# Simple sliding window on 1D tensor
x = torch.tensor([1, 2, 3, 4, 5, 6])
window_size = 3
stride = 1

# Using unfold
windows_unfold = x.unfold(0, window_size, stride)  # shape: (4, 3)
# Result: [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]

# Using advanced indexing
num_windows = (len(x) - window_size) // stride + 1
indices = torch.arange(window_size) + torch.arange(num_windows).unsqueeze(1) * stride
windows_index = x[indices]  # shape: (4, 3)
```

 ****

### Matrix Operations

| Operation | Syntax | Dimensions | Broadcasting | Use Case |
| --- | --- | --- | --- | --- |
| `torch.matmul` / `@` | `A @ B` | Flexible (≥2D) | Yes | General-purpose matrix multiplication |
| `torch.mm` | `torch.mm(A, B)` | 2D only | No | Efficient 2D matrix multiplication |
| `torch.bmm` | `torch.bmm(A, B)` | 3D only | No | Batched matrix multiplication |
| `*` | `A * B` | Any | Yes | Element-wise multiplication (Hadamard) |

**Advanced Matrix Multiplication:**

Matmul only looks at the last two dimensions. For other dimensions, use permute or einsum.

```python
# Example: A: [3, 4, 5, 6], B: [5, 7]
# Target: multiply A's dims [1,2] with B's dims [0,1]
# Expected output: [3, 4, 7, 6]

# Solution 1: Using permute
A_permuted = A.permute(0, 3, 1, 2)  # [3, 6, 4, 5]
result = A_permuted @ B             # [3, 6, 4, 7]
final_result = result.permute(0, 2, 3, 1)  # [3, 4, 7, 6]

# Solution 2: Using einsum (cleaner)
result = torch.einsum('abcd,de->abce', A, B)  # [3, 4, 7, 6]
```

### Common Tensor Functions

**Shape Manipulation:**

```python
# Permute - Reorders dimensions
x = torch.randn(2, 3, 4)
y = x.permute(2, 0, 1)      # shape becomes (4, 2, 3)

# Transpose - Swaps two dimensions
y = x.transpose(1, 2)       # shape becomes (2, 4, 3)

# Squeeze/Unsqueeze - Remove/Add singleton dimensions
x = torch.randn(1, 2, 1, 3)
squeezed = x.squeeze()      # Removes all singleton dimensions
unsqueezed = x.unsqueeze(1) # Adds dimension at position 1
```

**Combining Tensors:**

```python
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)

# Stack - Creates new dimension
torch.stack([x1, x2])           # (2, 2, 3)
torch.stack([x1, x2], dim=2)    # (2, 3, 2)

# Concatenate - Along existing dimension
torch.cat([x1, x2], dim=0)      # (4, 3)
torch.hstack([x1, x2])          # (2, 6) - horizontal stack
torch.vstack([x1, x2])          # (4, 3) - vertical stack
```

**Selection and Sorting:**

```python
# Conditional selection
condition = x > 0
result = torch.where(condition, x, torch.zeros_like(x))

# Max/Min with indices
values, indices = torch.max(x, dim=1)
min_values, min_indices = torch.min(x, dim=1)

# Sorting
sorted_values, sorted_indices = torch.sort(x, dim=1)
topk_values, topk_indices = torch.topk(x, k=3, dim=1)

# Argmax/Argmin
max_indices = torch.argmax(x, dim=1)
min_indices = torch.argmin(x, dim=1)
```

**Masked Operations:**

```python
mask = x > 0
masked_fill = x.masked_fill(mask, value=0)    # Fill where mask is True
masked_select = x.masked_select(mask)         # Returns 1D tensor
```

**Where**

```python
# Basic where usage
condition = x > 0
result = torch.where(condition, x, y)  # Returns x where condition is True, y where False

# Common patterns
zeros = torch.where(x > 0, x, torch.zeros_like(x))  # Zero out negative values
clipped = torch.where(x > threshold, threshold, x)   # Clip values above threshold

# Multiple conditions using nested where
result = torch.where(x > 0,
                    torch.where(x < 1, x, torch.ones_like(x)),  # If x > 0
                    torch.zeros_like(x))                        # If x <= 0

# Where with broadcasting
mask = x > 0  # shape: (batch_size, features)
scalar = torch.where(mask, x, 0.0)  # Broadcasting 0.0 to match x's shape

# Getting indices where condition is True
indices = torch.where(condition)  # Returns tuple of indices
selected = x[indices]  # Select values at those indices
```

### Folding and Unfolding Operations

```python
# Unfold - Splits dimension into multiple dimensions
x = torch.tensor([1, 2, 3, 4, 5, 6])  # shape: (6,)
unfolded = x.unfold(0, size=2, step=2)  # shape: (3, 2)
# Result: [[1, 2], [3, 4], [5, 6]]

# More complex unfold (2D) - Extract patches
matrix = torch.arange(16).reshape(4, 4)
patches = matrix.unfold(0, 2, 1).unfold(1, 2, 1)  # 2x2 patches with stride 1
# Result shape: (3, 3, 2, 2) - 3x3 grid of 2x2 patches

# im2col operation (common in CNN implementations)
def im2col(x, kernel_size):
    B, C, H, W = x.shape
    patches = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C * kernel_size * kernel_size)

# Rolling window using unfold
signal = torch.arange(10)
windows = signal.unfold(0, size=3, step=1)  # Overlapping windows of size 3
```

### Multiple Dimension Operations

**Dimension Support by Operation Type:**

- **Reduction operations** (sum, mean, std, var, norm): Usually support multiple dims
- **Indexing operations** (max, min, argmax, sort): Usually single dim only
- **Element-wise operations** (softmax, cumsum): Usually single dim only

```python
x = torch.randn(2, 3, 4, 5)

# Reduction operations - multiple dimensions supported
mean = torch.mean(x, dim=(1, 2))      # Reduces dimensions 1 and 2
sum_result = torch.sum(x, dim=(0, 3)) # Reduces dimensions 0 and 3

# Indexing operations - single dimension only
max_vals, _ = torch.max(x, dim=1)     # Single dimension
min_vals, _ = torch.min(x, dim=2)     # Single dimension

# Use torch.amax/torch.amin for multiple dimensions
max_multiple = torch.amax(x, dim=(1, 2))  # Multiple dimensions supported

# Normalization
normalized = torch.nn.functional.layer_norm(x, [4, 5])  # Normalize last 2 dims
```

```python
# Dimension operations
x = torch.randn(2, 3, 4, 5)

# Sum across dimensions
dim_sum = torch.sum(x, dim=1)          # Sum along dimension 1
multi_sum = torch.sum(x, dim=(1, 2))   # Sum along multiple dimensions

# Mean across dimensions
dim_mean = torch.mean(x, dim=2)        # Mean along dimension 2
multi_mean = torch.mean(x, dim=(0, 3)) # Mean along multiple dimensions

# Max/Min along dimensions
max_vals, max_idx = torch.max(x, dim=1)    # Maximum values and indices
min_vals, min_idx = torch.min(x, dim=2)    # Minimum values and indices

# Other dimension operations
std_dim = torch.std(x, dim=1)          # Standard deviation along dimension
var_dim = torch.var(x, dim=2)          # Variance along dimension
norm_dim = torch.norm(x, dim=3)        # L2 norm along dimension

# Cumulative operations
cumsum = torch.cumsum(x, dim=1)        # Cumulative sum along dimension
cumprod = torch.cumprod(x, dim=2)      # Cumulative product along dimension

# Argmax/Argmin along dimensions
argmax = torch.argmax(x, dim=1)        # Indices of maximum values
argmin = torch.argmin(x, dim=2)        # Indices of minimum values
```

---

# Einsum / Einops

The PyTorch einsum operation and the more general Einops library are very powerful tools that have been becoming popular recently.

## Einsum

Einstein notation has gotten very popular with transformers. Attention is much cleaner to implement with these operations, especially variants like multi head or group query attention.

PyTorch has a [torch.einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html#torch.einsum) operation. This lets you perform tensor operations using Einstein summation notation. The notation takes the format: 'input_indices, input_indices -> output_indices'. This enables cleaner code by reducing the need for reshaping and transposing dimensions. 

Einops operations are platform agnostic and work for NumPy, TensorFlow, PyTorch, and Jax, and also different device types (CPU/GPU/TPU). 

```python
# Basic matrix multiplication (2D x 2D)
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = torch.einsum('ij,jk->ik', a, b)  # Equivalent to a @ b or torch.matmul(a, b)

# Batch matrix multiplication (3D x 3D)
batch_a = torch.randn(10, 3, 4)
batch_b = torch.randn(10, 4, 5)
batch_c = torch.einsum('bij,bjk->bik', batch_a, batch_b)  # Batch matrix multiplication

# Dot product (1D x 1D)
v1 = torch.randn(5)
v2 = torch.randn(5)
dot = torch.einsum('i,i->', v1, v2)  # Equivalent to torch.dot(v1, v2)

# Outer product (1D x 1D)
outer = torch.einsum('i,j->ij', v1, v2)  # Equivalent to torch.outer(v1, v2)

# Batch dot product (2D x 2D)
batch_v1 = torch.randn(10, 5)
batch_v2 = torch.randn(10, 5)
batch_dot = torch.einsum('bi,bi->b', batch_v1, batch_v2)  # Dot product for each batch

# Diagonal (2D -> 1D)
matrix = torch.randn(3, 3)
diag = torch.einsum('ii->i', matrix)  # Equivalent to torch.diag(matrix, 0)

# Trace (2D -> scalar)
trace = torch.einsum('ii->', matrix)  # Equivalent to torch.trace(matrix)

# Transpose (2D -> 2D)
transpose = torch.einsum('ij->ji', matrix)  # Equivalent to matrix.T

```

We can implement more complicated functions with einstein notation.

```python
a = torch.randn(8, 4, 2, 3)
b = torch.randn(2, 3, 6)

# matrix multiplication on the last two dimensions of a and first two of b

# without einstein
a_flat = a.reshape(8, 4, -1) # or a.flatten(start_dim=2)
b_flat = b.reshape(-1, 6) # or b.flatten(end_dim=1)
c = a_flat@b_flat # (8, 4, 6)

# with einstein
c = torch.einsum('ijkl, klm -> ijm', a, b) # (8, 4, 6)

```

Let's look at how Multi-Head Attention (MHA) and Grouped Query Attention (GQA) can be implemented using einsum operations:

```python
# Group Query Attention implementation
# Parameters
batch_size = 2
seq_len = 10
d_model = 512
num_heads = 8
num_kv_heads = 2
head_dim = d_model // num_heads
heads_per_kv = num_heads // num_kv_heads

# Sample inputs
q = torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model)
v = torch.randn(batch_size, seq_len, d_model)

# Reshape
# Without Einstein
q_reshaped = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (batch, q_heads, seq, head_dim)
k_reshaped = k.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # (batch, kv_heads, seq, head_dim)
v_reshaped = v.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # (batch, kv_heads, seq, head_dim)

# With Einstein can skip transpose
q_reshaped = q.reshape(batch_size, seq_len, num_heads, head_dim)  # (batch, seq, heads, dim)
k_reshaped = k.reshape(batch_size, seq_len, num_kv_heads, head_dim)  # (batch, seq, kv_heads, dim)
v_reshaped = v.reshape(batch_size, seq_len, num_kv_heads, head_dim)  # (batch, seq, kv_heads, dim)

# Attention computation
# Without Einstein - requires explicit broadcasting/expansion (repeat_interleave)
k_expanded = k_reshaped.repeat_interleave(heads_per_kv, dim=1)  # (batch, q_heads, seq, head_dim)
v_expanded = v_reshaped.repeat_interleave(heads_per_kv, dim=1)  # (batch, q_heads, seq, head_dim)
scores = torch.matmul(q_reshaped, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
# (batch, q_heads, seq, seq)
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, v_expanded)  # (batch, q_heads, seq, head_dim)
output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model)

# With Einstein - automatic broadcasting, no explicit expansion needed
scores = torch.einsum('bshd,bSkd->bshS', q_reshaped, k_reshaped) / math.sqrt(head_dim)
# Automatic broadcasting: (batch, seq, q_heads, head_dim) × (batch, seq, kv_heads, head_dim) 
# -> (batch, seq, q_heads, seq) with implicit grouping
attn_weights = torch.softmax(scores, dim=-1)
output = torch.einsum('bshS,bSkd->bshd', attn_weights, v_reshaped)
# (batch, seq, q_heads, head_dim)
output = output.reshape(batch_size, seq_len, d_model)
```

## Einops

The [Einops](https://einops.rocks/) is a popular Python package that implements more operations with einstein notation.

There are a few important advantages

1. dimension names can be multiple characters which helps readability (”heads” instead of “h”)

### Rearrange

```python
b, c, h, w = 12, 3, 96, 96
x = torch.randn(b, c, h, w)

# rearrange
x.permute(0, 2, 3, 1)
rearrange(x, 'b c h w -> b h w c')
rearrange(x, 'batch channel height width -> batch height width channel')

# rearrange to collapsing dimensions
y = rearrange(x, 'b h w c -> (b h) w c')
# can rearrange and combine dimensions
rearrange(x, 'b c h w -> (b h) w c')

# decompose axis
x.reshape(3, 4, 3, 96, 96)
rearrange(x, '(b1 b2) c h w -> b1 b2 h w c', b1=3) # have to provide enough dim sizes to be able to infer the rest

# combine decomposition and composition
rearrange(x, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=3) # (3*96, 4*96, 3)

# can stretch dimensions
rearrange(x, "b h (w w2) c -> (h w2) (b w) c", w2=2) # half width and double height
rearrange(x, "b (h h2) w c -> (b h) (w h2) c", h2=2)

# before composing, we can reorder the dims
# this changes how the data is arranged
rearrange(x, "b h w c -> h (w b) c")
# PyTorch equivalent
result = ims.permute(1, 2, 0, 3)  # b h w c -> h w b c
result = result.reshape(h, w * b, c)  # h w b c -> h (w*b) c

# can also combine lists of tensors
batch_list = list(x)
# that's how we can stack inputs
# "list axis" becomes first ("b" in this case), and we left it there
rearrange(batch_list, "b h w c -> b h w c")
# reduces the need for stacking operations

# can add and squeeze dims
y = rearrange(x, "b h w c -> b 1 h w 1 c")  # functionality of numpy.expand_dims
rearrange(y, "b 1 h w 1 c -> b h w c")

# can replace concat / stack operations
rearrange(batch_list, "b c h w -> (b h) w c") # can concat on any dims

# can also unpack
r, g, b_channel = rearrange(x, 'b three h w -> three b h w')

# Can get use rearrange as a PyTorch Layer
from einops.layers.torch import Rearrange
Rearrange("b three h w -> three b h w")

```

### Reduce

```python
b, c, h, w = 12, 3, 96, 96
x = torch.randn(b, c, h, w)

reduce(x, "b h w c -> h w c", "mean")
torch.einsum("bhwc->hwc", x) / x.shape[0] # einsum can only do sums
# min, max, sum, prod also supported

# can reduce to singleton dimensions with ()
reduce(x, "b h w c -> b () () c", "mean")  # (b, 1, 1, c)
# can also use 1
reduce(x, "b h w c -> b 1 1 c", "mean")  # (b, 1, 1, c)

# Can implement max pooling
reduce(x, "b c (h 2) (w 2) -> b c h w", reduction="max")
# Easy to extend this to higher dimensions, more flexible than predefined pooling layers
# Also this approach is more flexible for differently shaped tensors

# Can get use reduce as a PyTorch Layer
from einops.layers.torch import Reduce
Reduce("b c (h 2) (w 2) -> b c h w", "max")
```

### Repeat

```python
b, c, h, w = 12, 3, 96, 96
x = torch.randn(b, c, h, w)

repeat(x, 'b c h w -> repeats b c h w', repeats=5)
repeat(x, 'b c h w -> 5 b c h w')

# repeat on axis, like repeat_interleave
repeat(x, 'b c h w -> b (2 c) h w') # repeat the whole dimension
repeat(x, 'b c h w -> b (c 2) h w') # repeat each position
```

### Einsum

Einops proves an einsum function that provides the benefits of the other functions.

```python
from einops import einsum

batch_size = 2
seq_len = 10
kv_seq_len = 15
d_model = 256
num_heads = 8
head_dim = d_model // num_heads  # 32

# Create inputs in d_model format, then reshape to create heads
query = torch.randn(batch_size, seq_len, d_model)     # [batch, seq, d_model]
key = torch.randn(batch_size, kv_seq_len, d_model)   # [batch, kv_seq, d_model]  
value = torch.randn(batch_size, kv_seq_len, d_model) # [batch, kv_seq, d_model]

# PyTorch
# Reshape to create heads
query_heads = query.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # [batch, heads, seq, dim]
key_head = key.reshape(batch_size, kv_seq_len, num_heads, head_dim).transpose(1, 2)       # [batch, heads, kv_seq, dim]
value_heads = value.reshape(batch_size, kv_seq_len, num_heads, head_dim).transpose(1, 2)   # [batch, heads, kv_seq, dim]

scores_torch = torch.einsum('bhsd,bhkd->bhsk', query_heads, key_heads) / math.sqrt(head_dim)
attn_torch = torch.softmax(scores_torch, dim=-1)
output_torch = torch.einsum('bhsk,bhkd->bhsd', attn_torch, value_heads)

# Einops
# Can create heads on the fly
scores_einops = einsum(
   query, key,
   'batch seq_q (heads dim), batch seq_k (heads dim) -> batch heads seq_q seq_k',
   heads=num_heads, dim=head_dim
) / math.sqrt(head_dim)
attn_einops = torch.softmax(scores_einops, dim=-1)
output_einops = einsum(
   attn_einops, value,
   'batch heads seq_q seq_k, batch seq_k (heads dim) -> batch heads seq_q dim',
   heads=num_heads, dim=head_dim
)

# We can also implement GQA
# keys and values take a lower dimension
key_kv = torch.randn(batch, kv_seq, kv_dim)
value_kv = torch.randn(batch, kv_seq, kv_dim)

# PyTorch GQA
query_gqa = query.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)          # [batch, 8, seq, dim]
key_gqa = key_kv.reshape(batch_size, kv_seq_len, num_kv_heads, head_dim).transpose(1, 2)     # [batch, 2, kv_seq, dim]
value_gqa = value_kv.reshape(batch_size, kv_seq_len, num_kv_heads, head_dim).transpose(1, 2) # [batch, 2, kv_seq, dim]

# Expand K,V to match Q heads
key_expanded = key_gqa.repeat_interleave(heads_per_kv, dim=1)    # [batch, 8, kv_seq, dim]
value_expanded = value_gqa.repeat_interleave(heads_per_kv, dim=1) # [batch, 8, kv_seq, dim]

scores_gqa_torch = torch.einsum('bhsd,bhkd->bhsk', query_gqa, key_expanded) / math.sqrt(head_dim)
attn_gqa_torch = torch.softmax(scores_gqa_torch, dim=-1)
output_gqa_torch = torch.einsum('bhsk,bhkd->bhsd', attn_gqa_torch, value_expanded)

# Einops GQA - automatic broadcasting, no manual expansion
scores_gqa_einops = einsum(
   query, key_kv,
   'batch seq_q (q_heads dim), batch seq_k (kv_heads dim) -> batch seq_q q_heads seq_k',
   q_heads=num_heads, kv_heads=num_kv_heads, dim=head_dim
) / math.sqrt(head_dim)
attn_gqa_einops = torch.softmax(scores_gqa_einops, dim=-1)
output_gqa_einops = einsum(
   attn_gqa_einops, value_kv,
   'batch seq_q q_heads seq_k, batch seq_k (kv_heads dim) -> batch seq_q q_heads dim',
   kv_heads=num_kv_heads, dim=head_dim
)
```

### Other Operations

```python
from einops import asnumpy
asnumpy(tensor) # converts tensors from any framework to NumPy and moves to CPU if needed

from einops import parse_shape
# can get a dict mapping dims to sizes
parse_shape(tensor, 'b c _ _') # _ skips the dims
# {'batch': 10, 'c': 32}
```

### EinMix

EinMix provides a way to apply transformations to specific dimensions of tensors using the Einstein notation. It's essentially a way to apply linear layers to specific dimensions of your tensor. It offers far more flexibility than default Linear layers.

It can be though of as an einsum operation that also is responsible for initializing weight matrices.

```python
from einops.layers.torch import EinMix

batch, sequence, channels, hidden_dim = 32, 20, 64, 128
x = torch.randn(batch, sequence, channels)

# Apply a linear transformation across the channel dimension
linear_layer = EinMix(
    'b s c -> b s d',
    weight_shape='c d',
    bias_shape='d',
    c=channels, d=hidden_dim
)
y = linear_layer(x)  # shape: [32, 20, 128]

# Einsum equivalent
weight, bias = torch.randn(channels, hidden_dim), torch.randn(hidden_dim
torch.einsum('bsc,ch->bsh', x, weight) + bias

# More complex example: Apply different transformations to different heads
heads, dim = 8, 32
x = torch.randn(batch, sequence, heads, dim)  # [batch, seq, heads, dim]

# Transform each head with its own weights
head_transform = EinMix(
    'b s h d -> b s h d_out',
    weight_shape='h d d_out',
    bias_shape='h d_out',
    h=heads, d=dim, d_out=64
)
y = head_transform(x)  # shape: [32, 20, 8, 64]
```

### Pack / Unpack

Previous operations required knowing all of the dimensions of the tensors. The pack and unpack operations also combining and splitting tensors with variable dimensions.

```python
from einops import pack, unpack

# Apply same operation to tensors with different batch dimensions
a = torch.randn(32, 128)        # [batch, features]
b = torch.randn(4, 8, 128)      # [batch1, batch2, features]
c = torch.randn(2, 3, 4, 128)   # [b1, b2, b3, features]
weight = torch.randn(128, 64)   # [features, out_features]

packed, ps = pack([a, b, c], "* features")
# ps stores the leading dimensions [32], [4, 8], [2, 3, 4]
# this is needed to unpack

result_packed = packed @ weight  # single matmul for all tensors
result_a, result_b, result_c = unpack(result_packed, ps, "* out_features")
# result shapes: [32, 64], [4, 8, 64], [2, 3, 4, 64]

# Universal function
def universal_predict(inputs):
    packed, ps = pack(inputs, "* features")
    return unpack(model(packed), ps, "* features")
```

---

# Model Creation

### Common Neural Network Layers

```python
# Linear layers
nn.Linear(in_features=784, out_features=128)
nn.LazyLinear(out_features=128)              # Infers input size

# Convolutional layers
nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3)

# Pooling layers
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2)
nn.AdaptiveAvgPool2d((7, 7))                 # Adaptive pooling

# Normalization layers
nn.BatchNorm1d(num_features=128)
nn.BatchNorm2d(num_features=64)
nn.LayerNorm(normalized_shape=[128])
nn.GroupNorm(num_groups=8, num_channels=64)

# Recurrent layers
nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
nn.GRU(input_size=100, hidden_size=128, batch_first=True)
nn.RNN(input_size=100, hidden_size=128, batch_first=True)

# Transformer layers
nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
)

# Activation functions
nn.ReLU()
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.LeakyReLU(negative_slope=0.01)
nn.ELU()
nn.Swish()  # or nn.SiLU()

# Regularization
nn.Dropout(p=0.5)
```

### Custom Modules and Layers

```python
import torch.nn as nn
import torch.nn.functional as F

# Basic custom module - parameter registration
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Register learnable parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)  # Explicitly register None
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# Complex module with submodules
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Submodules automatically registered
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

# Dynamic modules with ModuleList/ModuleDict
class FlexibleNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        # ModuleList for dynamic number of layers
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])
        # ModuleDict for named components
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        })
    
    def forward(self, x, activation='relu'):
        for layer in self.layers[:-1]:
            x = self.activations[activation](layer(x))
        return self.layers[-1](x)

# Sequential model - automatic module registration
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Advanced initialization patterns
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # Multiple parameter groups
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        
        # Register buffers (non-learnable tensors)
        self.register_buffer('scale', torch.tensor(dim ** -0.5))
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

# Usage examples
linear = CustomLinear(10, 5)
resnet_block = ResidualBlock(64)
flexible = FlexibleNet([784, 256, 128, 10])
attention = AttentionBlock(512, 8)

# Check registered parameters and modules
print("CustomLinear parameters:", list(linear.named_parameters()))
print("FlexibleNet modules:", list(flexible.named_modules()))
```

### Weight Initialization

```python
# Common initialization methods
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # Kaiming/He initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Apply initialization
model.apply(init_weights)

# Manual initialization
with torch.no_grad():
    model.linear.weight.fill_(0.1)
    model.linear.bias.zero_()

# Available initialization functions
nn.init.zeros_(tensor)
nn.init.ones_(tensor)
nn.init.normal_(tensor, mean=0, std=1)
nn.init.uniform_(tensor, a=0, b=1)
nn.init.xavier_uniform_(tensor)
nn.init.xavier_normal_(tensor)
nn.init.kaiming_uniform_(tensor)
nn.init.kaiming_normal_(tensor)
nn.init.orthogonal_(tensor)
```

### Module vs Functional Interface

**Module Interface (`torch.nn.Module`):**

- Object-oriented: defines layers as reusable objects
- Automatically registers parameters (weights, biases)
- Supports `.to()`, `.parameters()`, `.state_dict()`, etc.
- Used in model definitions (`nn.Conv1d`, `nn.Linear`)
- Integrates easily with optimizers and training loops

**Functional Interface (`torch.nn.functional`):**

- Stateless: performs operations without storing parameters
- Requires explicit input of weights and biases
- Useful for custom forward logic, weight sharing, or meta-learning
- Does not track state or integrate with model serialization
- Often used inside `forward()` for fine-grained control

### Loss Functions

```python
# Common loss functions
criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()

# Computing loss
loss = criterion(output, target)
```

### Optimization Functions

```python
# Basic optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
optimizer.zero_grad()    # Clear gradients
loss = criterion(output, target)
loss.backward()         # Compute gradients
optimizer.step()        # Update weights
```

### Basic Gradient Operations

```python
# Gradient computation
loss.backward()
grad = tensor.grad

# Gradient manipulation
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Detaching from computation graph
detached = tensor.detach()
no_grad = tensor.requires_grad_(False)

# Context manager for no gradient tracking
with torch.no_grad():
    # Computations here don't track gradients
    prediction = model(input)
```

---

# Training and Evaluation

```python
# Basic training loop
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

# Validation loop
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

# Shared sampling function
def sample_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    
    # Top-k filtering
    if top_k:
        logits[logits < torch.topk(logits, top_k)[0][:, [-1]]] = float('-inf')
    
    # Top-p filtering  
    if top_p:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        sorted_logits[mask] = float('-inf')
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    
    # Sample token
    return logits.argmax(dim=-1, keepdim=True) if temperature == 0 else \
           torch.multinomial(torch.softmax(logits, dim=-1), 1)

# Autoregressive inference (Transformer-style)
def generate(model, tokenizer, prompt, max_length=50, temperature=1.0, 
             top_k=None, top_p=None, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids).logits[:, -1, :]
            next_token = sample_token(logits, temperature, top_k, top_p)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id: break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# RNN-based generation (uses hidden state)
def generate_rnn(model, tokenizer, prompt, max_length=50, temperature=1.0, 
                 top_k=None, top_p=None, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    hidden = None
    
    with torch.no_grad():
        # Build hidden state from prompt
        for i in range(input_ids.size(1)):
            _, hidden = model(input_ids[:, i:i+1], hidden)
        
        # Generate tokens using hidden state
        current_token = input_ids[:, -1:]
        for _ in range(max_length):
            output, hidden = model(current_token, hidden)
            logits = output[:, -1, :]
            next_token = sample_token(logits, temperature, top_k, top_p)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            current_token = next_token
            if next_token.item() == tokenizer.eos_token_id: break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

---

# Data Handling

### Datasets and DataLoaders

**Dataset:**

- Abstract class that represents a dataset
- Must implement `__len__` and `__getitem__` methods
- Returns single samples when indexed

**DataLoader:**

- Wraps a Dataset and provides batching, shuffling, and parallel loading
- Controls batch size, sampling strategy, and worker processes
- Returns batches of samples as tensors

### Creating Custom Datasets

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

```

### Using DataLoader

```python
from torch.utils.data import DataLoader

dataset = MyDataset(data)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Iterating through batches
for batch in dataloader:
    # Process batch
    ...

```

### Data Transforms and Augmentation

```python
from torchvision import transforms

# Common transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Text transforms (if using torchtext)
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

# Custom transforms
class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Apply transforms
transformed_data = transform(image)

```

### Advanced Dataset Patterns

```python
# Dataset for image classification
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Dataset for sequence data
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, max_length=None):
        self.sequences = sequences
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]

        # Padding/truncation
        if self.max_length:
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            else:
                sequence = sequence + [0] * (self.max_length - len(sequence))

        return torch.tensor(sequence), torch.tensor(target)

# Custom collate function for variable-length sequences
def collate_fn(batch):
    sequences, targets = zip(*batch)

    # Pad sequences to max length in batch
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        padded = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded)

    return torch.tensor(padded_sequences), torch.tensor(targets)

# Use with DataLoader
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

```

---

# Model Management

### Saving and Loading Models

```python
# Save/load entire model (not recommended for production)
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# Save/load state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))

# Save/load with optimizer state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Loading checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Save for different devices
# Save on GPU, load on CPU
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# Partial loading (useful for transfer learning)
pretrained_dict = torch.load('pretrained.pth')
model_dict = model.state_dict()
# Filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

### Model Modes and Evaluation

```python
# Training vs evaluation modes
# Some model operations like dropout only occur during training mode
# Batch norm uses running estimates in eval model
model.train()
model.eval()

# Context manager to disable gradient calculation
with torch.no_grad():
    model.eval()
    predictions = model(test_data)

# Check current mode
print(model.training)  # True if in training mode

# Freeze/unfreeze parameters
for param in model.parameters():
    param.requires_grad = False  # Freeze all parameters

# Freeze specific layers
for param in model.feature_extractor.parameters():
    param.requires_grad = False

# Unfreeze specific layers
for param in model.classifier.parameters():
    param.requires_grad = True

```

---

# Advanced Modeling

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import *

# Built-in schedulers
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.95)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)

# Custom Learning Rate Scheduler
class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Generic custom scheduler - modify get_lr() for your needs"""
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        super(CustomScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Example: polynomial decay - modify this function for your schedule
        progress = self.last_epoch / self.total_epochs
        factor = (1 - progress) ** 0.9
        return [base_lr * factor for base_lr in self.base_lrs]

# Usage
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CustomScheduler(optimizer, total_epochs=100)
```

### Custom Autograd Functions

```python
# Define custom autograd function
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save tensors for backward pass
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Use custom function
relu = ReLUFunction.apply
output = relu(input_tensor)

# More complex example with multiple inputs/outputs
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Enable mixed precision
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Advanced Gradient Operations

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Gradient accumulation for large effective batch sizes
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Custom gradient computation
x = torch.randn(2, 2, requires_grad=True)
y = x.pow(2).sum()

# Compute gradients
grads = torch.autograd.grad(y, x, create_graph=True)  # For higher-order derivatives

# Manual gradient computation
x.grad = None  # Clear existing gradients
y.backward(retain_graph=True)  # Keep graph for multiple backward passes
print(x.grad)

# Gradient hooks for debugging
def print_grad(grad):
    print(f"Gradient: {grad}")

# Register hook on tensor
x.register_hook(print_grad)

# Register hook on module
def print_module_grad(module, grad_input, grad_output):
    print(f"Module: {module.__class__.__name__}")
    print(f"Grad output: {grad_output}")

model.conv1.register_backward_hook(print_module_grad)

# Freeze specific parameters
for name, param in model.named_parameters():
    if 'conv' in name:
        param.requires_grad = False
```

---

# Memory and Performance

### In-Place vs Out-of-Place Operations

| Aspect | In-Place Operations | Out-of-Place Operations |
| --- | --- | --- |
| **Syntax** | Suffix with underscore (`add_`, `mul_`) | No underscore (`add`, `mul`) |
| **Memory** | Modifies tensor directly | Creates new tensor |
| **Return** | Returns reference to modified tensor | Returns new tensor |
| **Autograd** | Can break computational graph | Preserves computational graph |

```python
# Out-of-place (creates new tensor)
a = torch.tensor([1, 2, 3])
b = a + 5       # b = [6, 7, 8], a is still [1, 2, 3]
c = a.add(5)    # c = [6, 7, 8], a is still [1, 2, 3]

# In-place (modifies existing tensor)
a = torch.tensor([1, 2, 3])
a += 5          # a = [6, 7, 8]
a.add_(5)       # a = [11, 12, 13]
```

**Important Notes:**

- In-place operations are not compatible with autograd
- Error: "a leaf Variable that requires grad has been used in an in-place operation"
- Use in-place operations for memory-efficient inference

### Memory Layout Operations

| Operation | Memory Copy | Contiguity Requirement | Memory Layout | Use Case |
| --- | --- | --- | --- | --- |
| **view()** | No | Yes | Shares storage, same stride | Fast reshaping when contiguous |
| **reshape()** | Maybe | No | Shares when possible, copies if needed | Safe general reshaping |
| **permute()** | No | No | Shares storage, changes stride | Reordering dimensions |
| **transpose()** | No | No | Shares storage, swaps dimensions | Special case of permute |

### Tensor Management Operations

| Operation | Type | Purpose | Returns New Tensor | Memory Copy |
| --- | --- | --- | --- | --- |
| `contiguous()` | Layout | Makes tensor memory-contiguous | Yes | Conditional |
| `clone()` | Copy | Deep copy of tensor | Yes | Yes |
| `detach()` | Graph | Detaches from computation graph | Yes | No |
| `to(device, dtype)` | Transfer | Moves/converts tensor | Yes | Conditional |
| `copy_(src)` | Copy | In-place copy from source | No | Yes |

```python
# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_gpu = tensor.to(device)
tensor_cpu = tensor.cpu()

# Type conversion
float_tensor = x.float()
long_tensor = x.long()

```

### In-Place Indexing Operations

```python
# index_add_ - Adds values at specified indices
tensor = torch.zeros(5, 3)
indices = torch.tensor([0, 2, 4])
source = torch.ones(3, 3)
tensor.index_add_(0, indices, source)  # Adds source rows at indices 0, 2, 4

# masked_fill_ - In-place filling where mask is True
tensor = torch.randn(3, 4)
mask = tensor > 0
tensor.masked_fill_(mask, value=0)     # Fills positive values with 0

# scatter_ - In-place scatter of values according to indices
tensor = torch.zeros(3, 5)
indices = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2], [2, 0, 1, 2, 0]])
values = torch.arange(15).reshape(3, 5)
tensor.scatter_(1, indices, values)    # Scatters values at indices in dim 1

# scatter_add_ - Accumulates values at indices
target = torch.zeros(3, 5)
target.scatter_add_(1, indices, values)
```

---

# Debugging and Optimization

### Memory Management

```python
# CUDA memory management
torch.cuda.empty_cache()                    # Free unused cached memory
torch.cuda.memory_allocated()               # Currently allocated memory
torch.cuda.memory_reserved()                # Currently reserved memory
torch.cuda.max_memory_allocated()           # Peak allocated memory

# Memory profiling
torch.cuda.memory_summary()                 # Detailed memory info
torch.cuda.reset_peak_memory_stats()        # Reset peak memory stats

# Monitor memory usage
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Gradient checkpointing for memory savings
from torch.utils.checkpoint import checkpoint

class MemoryEfficientBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)

    def forward(self, x):
        # Use checkpointing to trade compute for memory
        return checkpoint(self._forward_impl, x)

    def _forward_impl(self, x):
        return self.layer2(F.relu(self.layer1(x)))

```

### Performance Profiling

```python
# Basic profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your training code here
    model(inputs)

# Export profiling results
prof.export_chrome_trace("trace.json")

# Simple timing
import time
start_time = time.time()
# Your code here
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")

# CUDA timing (more accurate for GPU operations)
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# Your GPU code here
end_event.record()
torch.cuda.synchronize()

elapsed_time = start_event.elapsed_time(end_event)
print(f"GPU execution time: {elapsed_time:.4f} ms")

```

### Common Debugging Techniques

```python
# Check for NaN values
def check_for_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# Gradient checking
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            elif torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
            else:
                grad_norm = param.grad.norm()
                print(f"{name}: grad_norm = {grad_norm:.6f}")

# Debug model outputs
def debug_model_forward(model, input_tensor):
    hooks = []

    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__}: {output.shape}, mean: {output.mean():.4f}, std: {output.std():.4f}")
        if torch.isnan(output).any():
            print(f"NaN detected in {module.__class__.__name__}")

    # Register hooks on all modules
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    output = model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return output

# Tensor statistics
def tensor_stats(tensor, name="tensor"):
    print(f"{name} statistics:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min():.6f}")
    print(f"  Max: {tensor.max():.6f}")
    print(f"  Mean: {tensor.mean():.6f}")
    print(f"  Std: {tensor.std():.6f}")
    print(f"  Has NaN: {torch.isnan(tensor).any()}")
    print(f"  Has Inf: {torch.isinf(tensor).any()}")

# Model summary
def model_summary(model, input_size):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, *input_size)
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

# Set deterministic behavior for debugging
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Error Handling

```python
# Common error patterns and solutions
try:
    output = model(input_tensor)
except RuntimeError as e:
    error_msg = str(e)

    if "CUDA out of memory" in error_msg:
        print("GPU memory error. Try:")
        print("1. Reduce batch size")
        print("2. Use gradient checkpointing")
        print("3. Use mixed precision training")
        torch.cuda.empty_cache()

    elif "size mismatch" in error_msg:
        print("Tensor size mismatch. Check:")
        print("1. Input dimensions match model expectations")
        print("2. Batch dimensions are consistent")
        print(f"Input shape: {input_tensor.shape}")

    elif "device" in error_msg.lower():
        print("Device mismatch. Ensure all tensors are on same device:")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input device: {input_tensor.device}")

    raise e

# Automatic error recovery
class RobustTraining:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.best_state = None

    def save_checkpoint(self):
        self.best_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def restore_checkpoint(self):
        if self.best_state:
            self.model.load_state_dict(self.best_state['model'])
            self.optimizer.load_state_dict(self.best_state['optimizer'])

    def train_step(self, data, target):
        try:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Check for NaN gradients
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print("NaN gradient detected, skipping step")
                    return None

            self.optimizer.step()
            return loss.item()

        except RuntimeError as e:
            print(f"Training step failed: {e}")
            self.restore_checkpoint()
            return None
```

---

## Resources

### Practice and Learning

- [Tensor Puzzles](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb) - Tests broadcasting
- [Deep ML](https://www.deep-ml.com/) - ML coding practice
- [Tensor Gym](https://tensorgym.com/exercises) - Tensor operation exercises
- [NeetCode](https://neetcode.io/practice) - Coding practice
- [ML Coding Interviews](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLC/ml-coding.md) - Interview preparation