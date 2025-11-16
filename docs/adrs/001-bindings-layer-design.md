# ADR-001: Bindings Layer Design

**Status**: Accepted
**Date**: 2025-01-16
**Commit**: [To be filled]
**cuda-python Reference**: `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/utils.pyx`
**CUDA Version**: 1.0 (2007) - Runtime API Introduction

---

## Context

We need to expose CUDA C APIs to Python for our mini-cuda implementation. The full cuda-python repository uses **Cython** (`.pyx` files) for performance and low-level C integration. However, for a learning-focused implementation, we need to choose between:

1. **Cython**: Performance, type safety, direct C integration
2. **ctypes**: Pure Python, no compilation, simpler to understand
3. **cffi**: Middle ground, some compilation, C-like declarations

### Constraints

- **Learning Focus**: Code should be easy to read and understand
- **No Build Complexity**: Minimize compilation steps for students
- **Coverage**: Support essential CUDA Runtime APIs (~20 functions)
- **Platform**: Support Linux (Windows as secondary goal)
- **CUDA Version**: Target CUDA 11.x or 12.x

---

## Decision

We will use **ctypes** for the bindings layer in our mini-cuda implementation.

### Code Example

```python
# mini_cuda/bindings/runtime.py
import ctypes
import sys
import os

# Load CUDA Runtime library
def _load_cudart():
    if sys.platform.startswith('linux'):
        # Try system library first
        try:
            return ctypes.CDLL('libcudart.so')
        except OSError:
            # Try versioned libraries
            for ver in [12, 11, 10]:
                try:
                    return ctypes.CDLL(f'libcudart.so.{ver}')
                except OSError:
                    continue
            raise RuntimeError("Could not find CUDA Runtime library")
    elif sys.platform == 'win32':
        # Windows: cudart64_XX.dll
        cuda_path = os.environ.get('CUDA_PATH', '')
        if cuda_path:
            dll_path = os.path.join(cuda_path, 'bin', 'cudart64_12.dll')
            if os.path.exists(dll_path):
                return ctypes.CDLL(dll_path)
        raise RuntimeError("Could not find CUDA Runtime library on Windows")
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

_cudart = _load_cudart()

# Define CUDA types
cudaError_t = ctypes.c_int
cudaStream_t = ctypes.c_void_p
cudaEvent_t = ctypes.c_void_p

# Error codes
cudaSuccess = 0
cudaErrorMemoryAllocation = 2
cudaErrorInvalidValue = 11
cudaErrorNotReady = 600

# Wrap cudaMalloc
_cudaMalloc = _cudart.cudaMalloc
_cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudaMalloc.restype = cudaError_t

def cudaMalloc(size):
    """Allocate device memory

    Args:
        size: Size in bytes

    Returns:
        Device pointer (integer)

    Raises:
        CudaError: If allocation fails
    """
    ptr = ctypes.c_void_p()
    err = _cudaMalloc(ctypes.byref(ptr), size)
    if err != cudaSuccess:
        raise CudaError(err, "cudaMalloc failed")
    return ptr.value

# Wrap cudaFree
_cudaFree = _cudart.cudaFree
_cudaFree.argtypes = [ctypes.c_void_p]
_cudaFree.restype = cudaError_t

def cudaFree(ptr):
    """Free device memory

    Args:
        ptr: Device pointer

    Raises:
        CudaError: If free fails
    """
    err = _cudaFree(ctypes.c_void_p(ptr))
    if err != cudaSuccess:
        raise CudaError(err, "cudaFree failed")

# ... similar for other functions
```

---

## Rationale

### Why ctypes?

1. **No Compilation Required**
   - Pure Python implementation
   - No need for CUDA headers during installation
   - Students can edit and run immediately
   - Easier to experiment and debug

2. **Transparent C Mapping**
   - Direct mapping: `cudaMalloc` → `_cudart.cudaMalloc`
   - Clear distinction between C API and Python wrapper
   - Easy to reference CUDA C documentation

3. **Sufficient Performance**
   - ctypes overhead (~1-2µs per call) is negligible vs kernel launch time (~5-10µs)
   - Our focus is learning, not production performance
   - For comparison: cuda-python's Cython is ~10-20% faster, but adds complexity

4. **Platform Support**
   - Works on Linux and Windows without changes
   - Library loading is well-understood
   - No platform-specific build systems

### Alternatives Considered

#### Alternative A: Cython (like cuda-python)

**Why Rejected**:
- Requires compilation step
- Needs CUDA headers at build time
- `.pyx` syntax is less familiar than Python
- Harder to debug for beginners
- Overkill for 20-30 API functions

**When to Use**:
- Production code (cuda-python does this)
- Need maximum performance
- Calling thousands of CUDA APIs per second

**Example** (how cuda-python does it):
```cython
# cuda_bindings/cuda/bindings/_internal/utils.pyx
from libc.stdint cimport uintptr_t

cdef extern from "cuda_runtime_api.h":
    ctypedef int cudaError_t
    cudaError_t cudaMalloc(void** ptr, size_t size)

cpdef cudaMalloc_wrapper(size_t size):
    cdef void* ptr
    cdef cudaError_t err = cudaMalloc(&ptr, size)
    if err != 0:
        raise CudaError(err)
    return <uintptr_t>ptr
```

#### Alternative B: cffi

**Why Rejected**:
- Still requires compilation (JIT or AOT)
- More complex setup than ctypes
- Less common in educational contexts
- Not significantly better than ctypes for our use case

**When to Use**:
- Need better performance than ctypes
- Want to avoid full Cython complexity
- Have complex C structures to wrap

**Example**:
```python
from cffi import FFI
ffi = FFI()
ffi.cdef("int cudaMalloc(void** ptr, size_t size);")
lib = ffi.dlopen("libcudart.so")

def cudaMalloc(size):
    ptr = ffi.new("void**")
    err = lib.cudaMalloc(ptr, size)
    return int(ffi.cast("uintptr_t", ptr[0]))
```

---

## cuda-python Comparison

### What cuda-python Does

**File**: `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/utils.pyx` (lines 1-100)

cuda-python uses **Cython** for the bindings layer:

```cython
# Cython declaration (.pxd file)
cdef extern from "cuda_runtime_api.h":
    ctypedef int cudaError_t
    ctypedef void* cudaStream_t
    cudaError_t cudaMalloc(void** devPtr, size_t size)

# Cython implementation (.pyx file)
cpdef tuple cudaMalloc(size_t size):
    cdef void* ptr
    cdef cudaError_t err
    err = cudaMalloc(&ptr, size)
    return (err, <uintptr_t>ptr)
```

**Benefits of cuda-python's approach**:
- ✅ Maximum performance (~10-20% faster than ctypes)
- ✅ Type safety at compile time
- ✅ Direct C integration (no manual argtypes/restype)
- ✅ Better for wrapping hundreds of functions

**Drawbacks for learning**:
- ❌ Requires compilation step
- ❌ Needs CUDA headers at build time
- ❌ `.pyx` syntax less familiar
- ❌ Harder to debug for beginners

### What We're Doing Differently

**Our approach (ctypes)**:

```python
# Pure Python, no compilation
_cudaMalloc = _cudart.cudaMalloc
_cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudaMalloc.restype = cudaError_t

def cudaMalloc(size):
    ptr = ctypes.c_void_p()
    err = _cudaMalloc(ctypes.byref(ptr), size)
    if err != cudaSuccess:
        raise CudaError(err)
    return ptr.value
```

**Why it's better for learning**:
- ✅ No compilation required
- ✅ Pure Python (can edit and run immediately)
- ✅ Clear mapping to C API
- ✅ Easy to experiment with
- ✅ Sufficient performance for learning

**Trade-off**:
- ❌ Slightly slower (~1-2µs overhead per call)
- ❌ Manual argtypes/restype declarations
- ❌ No compile-time type checking

---

## Historical Evolution

### CUDA 1.0 (2007) - Runtime API Birth

**Original C API**:
```c
// cuda_runtime_api.h
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind);
```

**Design Philosophy**:
- Simple C API for basic operations
- Error codes (not exceptions)
- Explicit memory management
- Synchronous by default

### CUDA 2.0 (2008) - Asynchronous APIs

**New APIs**:
```c
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
```

**Impact on Bindings**:
- Need to wrap stream handles
- Async APIs require careful synchronization
- More complex error handling

### CUDA 12.x (2024) - Python Bindings (cuda-python)

**Python API Design**:
```python
# High-level (cuda.core)
buffer = DeviceBuffer(size)
buffer.copy_from_host(data, stream=s)

# Low-level (cuda.bindings)
from cuda.bindings import runtime
err, ptr = runtime.cudaMalloc(size)
```

**Philosophy**:
- **cuda.bindings**: Low-level, 1:1 mapping to C
- **cuda.core**: High-level, Pythonic abstractions
- **Our mini version**: Bridges both (learning-focused)

---

## Trade-offs

### Benefits

✅ **Simplicity**: Pure Python, no build system
✅ **Transparency**: Clear C ↔ Python mapping
✅ **Flexibility**: Easy to modify and experiment
✅ **Portability**: Works on any Python installation
✅ **Debugging**: Standard Python debugging tools
✅ **Learning**: Focus on CUDA concepts, not Cython syntax

### Limitations

❌ **Performance**: ~10-20% slower than Cython (negligible for learning)
❌ **Type Safety**: Runtime errors vs compile-time
❌ **Scaling**: Manual work for each function (but we only need ~20)
❌ **Advanced Features**: No Cython-specific optimizations

---

## Learning Outcomes

After this commit, you should understand:

1. **ctypes Basics**
   - How to load shared libraries (`CDLL`)
   - How to declare function signatures (`argtypes`, `restype`)
   - How to convert between Python and C types

2. **CUDA Runtime API Structure**
   - Error codes vs exceptions
   - Pointer-based outputs (`void**`)
   - Platform-specific library loading

3. **Design Trade-offs**
   - Performance vs Simplicity
   - Compilation vs Pure Python
   - When to use ctypes vs Cython vs cffi

4. **cuda-python Architecture**
   - Why they chose Cython (production performance)
   - How bindings layer relates to high-level API
   - Two-layer design pattern (bindings + core)

---

## References

### cuda-python Source
- **Bindings**: `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/utils.pyx`
- **Linux Implementation**: `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/nvvm_linux.pyx`
- **Windows Implementation**: `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/nvvm_windows.pyx`

### CUDA Documentation
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Python Documentation
- [ctypes — A foreign function library](https://docs.python.org/3/library/ctypes.html)
- [Cython Documentation](https://cython.readthedocs.io/)

### Academic Papers
- "CUDA: Scalable Parallel Programming for General-Purpose GPU Computing" (Nickolls et al., 2008)

---

## Exercises

### Exercise 1: Extend Bindings
**Task**: Add `cudaMemset` to the bindings layer

**Hints**:
- C signature: `cudaError_t cudaMemset(void *devPtr, int value, size_t count)`
- Use similar pattern to `cudaMalloc`
- Test with a simple example

**Expected Code**:
```python
_cudaMemset = _cudart.cudaMemset
_cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_cudaMemset.restype = cudaError_t

def cudaMemset(ptr, value, count):
    """Set device memory to value"""
    err = _cudaMemset(ctypes.c_void_p(ptr), value, count)
    if err != cudaSuccess:
        raise CudaError(err, "cudaMemset failed")
```

### Exercise 2: Compare Performance
**Task**: Measure ctypes overhead

**Approach**:
1. Call `cudaMalloc` 10,000 times
2. Measure total time
3. Calculate per-call overhead
4. Compare to kernel launch time (~5-10µs)

**Expected Result**: ~1-2µs per call (negligible)

### Exercise 3: Read cuda-python's Cython
**Task**: Study `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/utils.pyx`

**Questions**:
1. How does Cython declare C functions?
2. What is `cdef extern from`?
3. How does Cython handle error codes?
4. Why use `cpdef` instead of `def`?

---

## Next Steps

After completing this ADR:
1. ✅ Implement `mini_cuda/bindings/runtime.py`
2. ✅ Add unit tests for basic bindings
3. ⏭️ Move to **ADR-002**: Memory Management Design
4. 📚 Read cuda-python's bindings implementation for comparison

---

**Status**: Ready to implement
**Confidence**: High (well-established pattern)
**Risk**: Low (ctypes is stable and well-documented)
