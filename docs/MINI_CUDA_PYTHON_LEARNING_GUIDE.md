# Mini CUDA-Python Learning Guide
## Historically-Grounded, Reasoning-Based Implementation Framework

> **Based on cuda-python codebase analysis**
> **Goal**: Learn GPU programming & CUDA by building a mini version from scratch

---

## 📊 Quick Reference

### What's in cuda-python (Full Repository)
- **Total Components**: 4 main packages (cuda.core, cuda.bindings, cuda.pathfinder, cuda.python)
- **Core Modules**: ~15 major components (Device, Stream, Event, Memory, Kernel, etc.)
- **Bindings Layer**: Low-level Cython bindings to CUDA C APIs
- **High-level Layer**: Pythonic wrappers (cuda.core.experimental)
- **Total Code**: ~50,000+ lines (bindings + core + tests)

### What Your Mini Version Should Have
- **Target Size**: 3,000-5,000 lines of Python/Cython
- **Core Components**: 8-10 essential modules
- **Bindings**: Simple ctypes/cffi wrappers (no full Cython initially)
- **High-level API**: Pythonic interface for 5-6 core operations
- **Memory**: Basic device allocation/deallocation
- **Execution**: Simple kernel launch, streams, events

---

## 🏗️ Architecture Overview

```
Python User Code
         ↓
  CUDA.CORE (High-level)      (1,500-2,000 lines)
  ├─ Device Management
  ├─ Memory Management
  ├─ Kernel Launch
  ├─ Stream Management
  └─ Event Synchronization
         ↓
  CUDA.BINDINGS (Low-level)   (1,000-1,500 lines)
  ├─ ctypes/cffi wrappers
  ├─ Error handling
  └─ API mapping
         ↓
  CUDA Driver/Runtime API (C)
         ↓
     GPU Hardware
```

---

## 📚 Component Design Details

### 1. CUDA BINDINGS LAYER (1,000-1,500 lines)

**Purpose**: Low-level Python bindings to CUDA C APIs

**What cuda-python does**:
- Uses Cython for performance (.pyx files)
- Supports CUDA Driver, Runtime, NVRTC, NVVM, cuFile
- Platform-specific implementations (Linux/Windows)
- Reference: `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/`

**What mini version should do**:
- Use `ctypes` or `cffi` (simpler than Cython for learning)
- Focus on CUDA Runtime API only (skip Driver API for MVP)
- Single platform (Linux or Windows)
- 10-15 essential API functions

**Essential CUDA Runtime APIs to wrap**:

```python
# Device Management (3 functions)
cudaGetDeviceCount()
cudaGetDeviceProperties()
cudaSetDevice()

# Memory Management (5 functions)
cudaMalloc()
cudaFree()
cudaMemcpy()
cudaMemcpyAsync()
cudaMemset()

# Stream Management (4 functions)
cudaStreamCreate()
cudaStreamDestroy()
cudaStreamSynchronize()
cudaStreamQuery()

# Event Management (5 functions)
cudaEventCreate()
cudaEventDestroy()
cudaEventRecord()
cudaEventSynchronize()
cudaEventElapsedTime()

# Kernel Launch (1 function)
cudaLaunchKernel()

# Synchronization (2 functions)
cudaDeviceSynchronize()
cudaDeviceReset()

# Error Handling (2 functions)
cudaGetLastError()
cudaGetErrorString()
```

**Simplified Implementation**:

```python
# mini_cuda/bindings/runtime.py
import ctypes
import sys

# Load CUDA Runtime library
if sys.platform == 'linux':
    _cudart = ctypes.CDLL('libcudart.so')
elif sys.platform == 'win32':
    _cudart = ctypes.CDLL('cudart64_XX.dll')  # XX = CUDA version

# Define CUDA types
cudaError_t = ctypes.c_int
cudaStream_t = ctypes.c_void_p
cudaEvent_t = ctypes.c_void_p

# Wrap cudaMalloc
_cudaMalloc = _cudart.cudaMalloc
_cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudaMalloc.restype = cudaError_t

def cudaMalloc(size):
    """Allocate device memory"""
    ptr = ctypes.c_void_p()
    err = _cudaMalloc(ctypes.byref(ptr), size)
    if err != 0:
        raise CudaError(err)
    return ptr.value

# ... similar for other functions
```

**File Organization**:
```
mini_cuda/
├── bindings/
│   ├── __init__.py
│   ├── runtime.py      (CUDA Runtime API - 400 lines)
│   ├── types.py        (CUDA types & enums - 200 lines)
│   ├── errors.py       (Error handling - 100 lines)
│   └── loader.py       (Library loading - 100 lines)
```

**Reference**:
- `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/utils.pyx`
- CPython ctypes documentation

---

### 2. DEVICE MANAGEMENT (300-400 lines)

**Purpose**: Manage GPU devices, query properties

**What cuda-python does**:
- Full device property exposure
- Multi-GPU support
- Context management (implicit/explicit)
- Reference: `/home/user/cuda-python/cuda_core/cuda/core/experimental/_device.pyx` (56k lines!)

**What mini version should do**:
- Simple Device class
- Basic property queries (name, compute capability, memory)
- Single GPU focus (multi-GPU in Phase 2)

**Implementation**:

```python
# mini_cuda/core/device.py
from mini_cuda.bindings import runtime

class Device:
    """Represents a CUDA-capable GPU device"""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self._properties = None

    @staticmethod
    def count():
        """Return number of CUDA devices"""
        return runtime.cudaGetDeviceCount()

    def set_current(self):
        """Set this device as current"""
        runtime.cudaSetDevice(self.device_id)

    @property
    def name(self):
        """Device name"""
        if self._properties is None:
            self._properties = runtime.cudaGetDeviceProperties(self.device_id)
        return self._properties.name

    @property
    def compute_capability(self):
        """Compute capability (major, minor)"""
        if self._properties is None:
            self._properties = runtime.cudaGetDeviceProperties(self.device_id)
        return (self._properties.major, self._properties.minor)

    @property
    def total_memory(self):
        """Total device memory in bytes"""
        if self._properties is None:
            self._properties = runtime.cudaGetDeviceProperties(self.device_id)
        return self._properties.totalGlobalMem

    def __repr__(self):
        return f"Device({self.device_id}, {self.name})"
```

**Key Design Decisions**:
- **Lazy property loading**: Only query properties when accessed
- **Explicit device selection**: User calls `set_current()` vs automatic context switching
- **Single device object**: One Device per GPU vs shared singleton

**Reference**:
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_device.pyx`

---

### 3. MEMORY MANAGEMENT (400-600 lines)

**Purpose**: Allocate/free device memory, transfer data

**What cuda-python does**:
- Multiple memory types (device, host, pinned, managed)
- Memory pools & async allocation
- Virtual memory management
- Reference: `/home/user/cuda-python/cuda_core/cuda/core/experimental/_memory/`

**What mini version should do**:
- Simple device memory allocation
- Host ↔ Device transfers
- Basic memory buffer class
- Skip pools/virtual memory for MVP

**Implementation**:

```python
# mini_cuda/core/memory.py
from mini_cuda.bindings import runtime
import numpy as np

class DeviceBuffer:
    """Device memory buffer"""

    def __init__(self, size, device=None):
        """Allocate device memory

        Args:
            size: Size in bytes
            device: Device to allocate on (default: current)
        """
        self.size = size
        self.device = device or Device.current()
        self.device.set_current()

        # Allocate device memory
        self.ptr = runtime.cudaMalloc(size)

    def copy_from_host(self, host_array, stream=None):
        """Copy data from host to device

        Args:
            host_array: numpy array or bytes
            stream: Optional CUDA stream for async copy
        """
        if isinstance(host_array, np.ndarray):
            host_ptr = host_array.ctypes.data
            size = host_array.nbytes
        else:
            host_ptr = host_array
            size = len(host_array)

        if size > self.size:
            raise ValueError(f"Data size {size} exceeds buffer size {self.size}")

        if stream is None:
            runtime.cudaMemcpy(self.ptr, host_ptr, size,
                             runtime.cudaMemcpyHostToDevice)
        else:
            runtime.cudaMemcpyAsync(self.ptr, host_ptr, size,
                                   runtime.cudaMemcpyHostToDevice, stream.handle)

    def copy_to_host(self, host_array=None, stream=None):
        """Copy data from device to host

        Args:
            host_array: Optional pre-allocated numpy array
            stream: Optional CUDA stream for async copy

        Returns:
            numpy array with data
        """
        if host_array is None:
            host_array = np.empty(self.size, dtype=np.uint8)

        host_ptr = host_array.ctypes.data
        size = min(host_array.nbytes, self.size)

        if stream is None:
            runtime.cudaMemcpy(host_ptr, self.ptr, size,
                             runtime.cudaMemcpyDeviceToHost)
        else:
            runtime.cudaMemcpyAsync(host_ptr, self.ptr, size,
                                   runtime.cudaMemcpyDeviceToHost, stream.handle)

        return host_array

    def __del__(self):
        """Free device memory on deletion"""
        if hasattr(self, 'ptr') and self.ptr:
            runtime.cudaFree(self.ptr)

    def __repr__(self):
        return f"DeviceBuffer(size={self.size}, ptr=0x{self.ptr:x})"
```

**Key Design Decisions**:
- **RAII pattern**: Automatic cleanup via `__del__`
- **Numpy integration**: Direct support for numpy arrays
- **Sync vs Async**: Support both synchronous and stream-based async copies
- **Size validation**: Check buffer overflow before copy

**Reference**:
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_memory/__init__.py`
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_memoryview.pyx`

---

### 4. STREAM MANAGEMENT (200-300 lines)

**Purpose**: Manage asynchronous execution streams

**What cuda-python does**:
- Stream creation with flags
- Stream priorities
- Stream callbacks
- Reference: `/home/user/cuda-python/cuda_core/cuda/core/experimental/_stream.pyx`

**What mini version should do**:
- Basic stream creation/destruction
- Stream synchronization
- Default stream support

**Implementation**:

```python
# mini_cuda/core/stream.py
from mini_cuda.bindings import runtime

class Stream:
    """CUDA stream for asynchronous operations"""

    def __init__(self, non_blocking=True):
        """Create a CUDA stream

        Args:
            non_blocking: If True, create non-blocking stream
        """
        flags = runtime.cudaStreamNonBlocking if non_blocking else 0
        self.handle = runtime.cudaStreamCreate(flags)
        self.non_blocking = non_blocking

    def synchronize(self):
        """Wait for all operations in stream to complete"""
        runtime.cudaStreamSynchronize(self.handle)

    def query(self):
        """Check if all operations in stream are complete

        Returns:
            True if complete, False otherwise
        """
        err = runtime.cudaStreamQuery(self.handle)
        if err == runtime.cudaSuccess:
            return True
        elif err == runtime.cudaErrorNotReady:
            return False
        else:
            raise CudaError(err)

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.synchronize()
        return False

    def __del__(self):
        """Destroy stream on deletion"""
        if hasattr(self, 'handle') and self.handle:
            runtime.cudaStreamDestroy(self.handle)

    def __repr__(self):
        return f"Stream(handle=0x{self.handle:x}, non_blocking={self.non_blocking})"
```

**Key Design Decisions**:
- **Context manager support**: Enable `with Stream() as s:` pattern
- **Non-blocking default**: Better performance for concurrent operations
- **Query vs Synchronize**: Polling vs blocking wait

**Reference**:
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_stream.pyx`

---

### 5. EVENT MANAGEMENT (200-300 lines)

**Purpose**: Synchronization and timing

**What cuda-python does**:
- Event creation with flags
- Event recording on streams
- Event elapsed time calculation
- Reference: `/home/user/cuda-python/cuda_core/cuda/core/experimental/_event.pyx`

**What mini version should do**:
- Basic event creation/recording
- Synchronization
- Elapsed time measurement

**Implementation**:

```python
# mini_cuda/core/event.py
from mini_cuda.bindings import runtime

class Event:
    """CUDA event for synchronization and timing"""

    def __init__(self, enable_timing=True, blocking_sync=False):
        """Create a CUDA event

        Args:
            enable_timing: If True, event can be used for timing
            blocking_sync: If True, sync uses blocking wait
        """
        flags = 0
        if not enable_timing:
            flags |= runtime.cudaEventDisableTiming
        if blocking_sync:
            flags |= runtime.cudaEventBlockingSync

        self.handle = runtime.cudaEventCreate(flags)
        self.enable_timing = enable_timing

    def record(self, stream=None):
        """Record event on stream

        Args:
            stream: Stream to record on (default: default stream)
        """
        stream_handle = stream.handle if stream else 0
        runtime.cudaEventRecord(self.handle, stream_handle)

    def synchronize(self):
        """Wait for event to complete"""
        runtime.cudaEventSynchronize(self.handle)

    def elapsed_time(self, end_event):
        """Calculate elapsed time to another event

        Args:
            end_event: End event

        Returns:
            Elapsed time in milliseconds
        """
        if not self.enable_timing or not end_event.enable_timing:
            raise ValueError("Both events must have timing enabled")

        return runtime.cudaEventElapsedTime(self.handle, end_event.handle)

    def __del__(self):
        """Destroy event on deletion"""
        if hasattr(self, 'handle') and self.handle:
            runtime.cudaEventDestroy(self.handle)

    def __repr__(self):
        return f"Event(handle=0x{self.handle:x}, timing={self.enable_timing})"
```

**Key Design Decisions**:
- **Timing by default**: Most use cases need timing
- **Stream association**: Events are recorded on streams
- **Elapsed time**: Simple API for benchmarking

**Reference**:
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_event.pyx`

---

### 6. KERNEL LAUNCH (300-500 lines)

**Purpose**: Launch CUDA kernels from Python

**What cuda-python does**:
- Full kernel parameter handling
- Grid/block configuration
- Shared memory specification
- Reference: `/home/user/cuda-python/cuda_core/cuda/core/experimental/_launcher.pyx`

**What mini version should do**:
- Simple kernel launch API
- Basic parameter packing
- Grid/block dimensions

**Implementation**:

```python
# mini_cuda/core/kernel.py
from mini_cuda.bindings import runtime
import ctypes

class LaunchConfig:
    """Kernel launch configuration"""

    def __init__(self, grid, block, shared_mem=0, stream=None):
        """Create launch configuration

        Args:
            grid: Grid dimensions (int or tuple of 1-3 ints)
            block: Block dimensions (int or tuple of 1-3 ints)
            shared_mem: Shared memory size in bytes
            stream: Optional stream to launch on
        """
        self.grid = self._to_dim3(grid)
        self.block = self._to_dim3(block)
        self.shared_mem = shared_mem
        self.stream = stream

    @staticmethod
    def _to_dim3(value):
        """Convert int or tuple to (x, y, z) tuple"""
        if isinstance(value, int):
            return (value, 1, 1)
        elif len(value) == 1:
            return (value[0], 1, 1)
        elif len(value) == 2:
            return (value[0], value[1], 1)
        elif len(value) == 3:
            return tuple(value)
        else:
            raise ValueError("Dimension must be int or tuple of 1-3 ints")

    def __repr__(self):
        return f"LaunchConfig(grid={self.grid}, block={self.block})"


class Kernel:
    """CUDA kernel wrapper"""

    def __init__(self, func_ptr, name="kernel"):
        """Create kernel from function pointer

        Args:
            func_ptr: CUDA device function pointer
            name: Kernel name for debugging
        """
        self.func_ptr = func_ptr
        self.name = name

    def launch(self, config, *args):
        """Launch kernel

        Args:
            config: LaunchConfig object
            *args: Kernel arguments (must be ctypes or DeviceBuffer)
        """
        # Pack arguments
        kernel_args = []
        for arg in args:
            if isinstance(arg, DeviceBuffer):
                # Device buffer → pass pointer
                kernel_args.append(ctypes.c_void_p(arg.ptr))
            elif isinstance(arg, (int, float)):
                # Scalar → pass by value
                if isinstance(arg, int):
                    kernel_args.append(ctypes.c_int(arg))
                else:
                    kernel_args.append(ctypes.c_float(arg))
            else:
                # Assume ctypes object
                kernel_args.append(arg)

        # Create argument array
        args_array = (ctypes.c_void_p * len(kernel_args))()
        for i, arg in enumerate(kernel_args):
            args_array[i] = ctypes.addressof(arg)

        # Launch kernel
        stream_handle = config.stream.handle if config.stream else 0
        runtime.cudaLaunchKernel(
            self.func_ptr,
            config.grid, config.block,
            args_array, len(kernel_args),
            config.shared_mem,
            stream_handle
        )

    def __call__(self, config, *args):
        """Convenience: kernel(config, args)"""
        self.launch(config, *args)

    def __repr__(self):
        return f"Kernel({self.name})"
```

**Key Design Decisions**:
- **Separate config**: LaunchConfig object vs inline parameters
- **Argument packing**: Automatic conversion of Python types
- **Callable interface**: Support `kernel(config, args)` syntax

**Reference**:
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_launcher.pyx`
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_kernel_arg_handler.pyx`

---

### 7. MODULE & PROGRAM (400-600 lines)

**Purpose**: Load and manage compiled CUDA code

**What cuda-python does**:
- Program compilation from source (NVRTC)
- Module loading from PTX/CUBIN
- Kernel function lookup
- Reference: `/home/user/cuda-python/cuda_core/cuda/core/experimental/_module.py`

**What mini version should do**:
- Load pre-compiled PTX/CUBIN
- Extract kernel functions
- Skip runtime compilation for MVP

**Implementation**:

```python
# mini_cuda/core/module.py
from mini_cuda.bindings import driver  # Need Driver API for module loading
from mini_cuda.core.kernel import Kernel

class Module:
    """CUDA module (loaded PTX/CUBIN)"""

    def __init__(self, ptx_or_cubin, module_type='ptx'):
        """Load CUDA module

        Args:
            ptx_or_cubin: PTX string or CUBIN bytes
            module_type: 'ptx' or 'cubin'
        """
        self.module_type = module_type

        if module_type == 'ptx':
            self.handle = driver.cuModuleLoadData(ptx_or_cubin.encode())
        else:
            self.handle = driver.cuModuleLoadData(ptx_or_cubin)

        self._kernels = {}

    def get_kernel(self, name):
        """Get kernel function by name

        Args:
            name: Kernel function name

        Returns:
            Kernel object
        """
        if name in self._kernels:
            return self._kernels[name]

        func_ptr = driver.cuModuleGetFunction(self.handle, name.encode())
        kernel = Kernel(func_ptr, name)
        self._kernels[name] = kernel
        return kernel

    def __getattr__(self, name):
        """Convenience: module.kernel_name"""
        return self.get_kernel(name)

    def __del__(self):
        """Unload module on deletion"""
        if hasattr(self, 'handle') and self.handle:
            driver.cuModuleUnload(self.handle)

    def __repr__(self):
        return f"Module(type={self.module_type}, kernels={list(self._kernels.keys())})"
```

**Key Design Decisions**:
- **Pre-compiled code**: Load PTX/CUBIN vs runtime compilation
- **Lazy kernel loading**: Only load kernels when accessed
- **Attribute access**: `module.kernel_name` convenience

**Reference**:
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_module.py`

---

## 📅 Phased Implementation Timeline with Historical Context

### Historical Context: CUDA Evolution

| Year | CUDA Version | Major Features | Relevance |
|------|--------------|----------------|-----------|
| 2007 | CUDA 1.0 | First release, basic kernel launch | Foundation |
| 2009 | CUDA 2.0 | Streams, zero-copy | Async execution |
| 2010 | CUDA 3.0 | Unified addressing, CUDA C++ | Memory model |
| 2012 | CUDA 5.0 | Dynamic parallelism | Advanced features |
| 2016 | CUDA 8.0 | Unified memory, nvJitLink | Modern CUDA |
| 2020 | CUDA 11.0 | Cooperative groups, graphs | Current paradigm |
| 2024 | CUDA 12.x | Enhanced features | Latest |

---

## Phase 1: Bindings Layer (Est. 3-4 commits, ~1,000 lines)

**Historical Context**: CUDA 1.0 (2007) - Basic Runtime API

| Commit | Feature | ADR | cuda-python Ref | CUDA History | Lines | Learning Focus |
|--------|---------|-----|-----------------|--------------|-------|----------------|
| 1.1 | Project setup + library loading | ADR-001 | cuda_bindings/cuda/bindings/_internal/utils.pyx | CUDA 1.0 basics | 200 | ctypes/cffi, shared library loading |
| 1.2 | Device & memory APIs | ADR-002 | cuda_bindings/cuda/bindings/_internal/ | CUDA memory model | 300 | cudaMalloc, cudaMemcpy patterns |
| 1.3 | Stream & event APIs | ADR-003 | cuda_bindings/cuda/bindings/_internal/ | CUDA 2.0 async model | 250 | Asynchronous execution |
| 1.4 | Error handling & types | ADR-004 | cuda_bindings/cuda/bindings/_internal/utils.pyx | CUDA error codes | 250 | Error propagation |

**Checkpoint 1**: Allocate device memory, copy data, and synchronize

---

## Phase 2: Core Device & Memory (Est. 4 commits, ~1,000 lines)

**Historical Context**: CUDA 3.0 (2010) - Unified addressing

| Commit | Feature | ADR | cuda-python Ref | CUDA History | Lines | Learning Focus |
|--------|---------|-----|-----------------|--------------|-------|----------------|
| 2.1 | Device class | ADR-005 | cuda_core/experimental/_device.pyx | Multi-GPU management | 300 | Device properties, context |
| 2.2 | DeviceBuffer class | ADR-006 | cuda_core/experimental/_memory/ | Memory abstractions | 400 | RAII, numpy integration |
| 2.3 | Memory transfers | ADR-007 | cuda_core/experimental/_memory/ | H2D/D2H patterns | 200 | Sync vs async copies |
| 2.4 | Memory utilities | ADR-008 | cuda_core/experimental/_memoryview.pyx | Memory views | 100 | Buffer protocol |

**Checkpoint 2**: Create buffers, transfer numpy arrays to/from GPU

---

## Phase 3: Streams & Events (Est. 3 commits, ~600 lines)

**Historical Context**: CUDA 2.0 (2009) - Streams for async execution

| Commit | Feature | ADR | cuda-python Ref | CUDA History | Lines | Learning Focus |
|--------|---------|-----|-----------------|--------------|-------|----------------|
| 3.1 | Stream class | ADR-009 | cuda_core/experimental/_stream.pyx | Stream creation/sync | 250 | Async execution model |
| 3.2 | Event class | ADR-010 | cuda_core/experimental/_event.pyx | Timing & sync | 250 | Event-based synchronization |
| 3.3 | Stream operations | ADR-011 | cuda_core/experimental/_stream.pyx | Stream priority, callbacks | 100 | Advanced stream features |

**Checkpoint 3**: Overlap H2D copy with kernel execution using streams

---

## Phase 4: Kernel Launch (Est. 4 commits, ~800 lines)

**Historical Context**: CUDA 1.0 (2007) - Basic kernel launch

| Commit | Feature | ADR | cuda-python Ref | CUDA History | Lines | Learning Focus |
|--------|---------|-----|-----------------|--------------|-------|----------------|
| 4.1 | LaunchConfig class | ADR-012 | cuda_core/experimental/_launch_config.pyx | Grid/block config | 150 | Execution configuration |
| 4.2 | Kernel class | ADR-013 | cuda_core/experimental/_module.py | Kernel abstraction | 200 | Function pointers |
| 4.3 | Argument packing | ADR-014 | cuda_core/experimental/_kernel_arg_handler.pyx | Type conversion | 300 | Parameter marshaling |
| 4.4 | Kernel launch | ADR-015 | cuda_core/experimental/_launcher.pyx | cudaLaunchKernel | 150 | Kernel invocation |

**Checkpoint 4**: Load PTX and launch a simple vector addition kernel

---

## Phase 5: Module Loading (Est. 3 commits, ~600 lines)

**Historical Context**: CUDA 1.0 (2007) - Module loading from PTX

| Commit | Feature | ADR | cuda-python Ref | CUDA History | Lines | Learning Focus |
|--------|---------|-----|-----------------|--------------|-------|----------------|
| 5.1 | Module class | ADR-016 | cuda_core/experimental/_module.py | PTX/CUBIN loading | 300 | cuModuleLoad* |
| 5.2 | Kernel extraction | ADR-017 | cuda_core/experimental/_module.py | Function lookup | 200 | cuModuleGetFunction |
| 5.3 | Module utilities | ADR-018 | cuda_core/experimental/_module.py | Helper functions | 100 | Module management |

**Checkpoint 5**: Load a module, extract multiple kernels, launch them

---

## Phase 6: Integration & Examples (Est. 3 commits, ~500 lines)

**Historical Context**: Modern CUDA (2024) - Pythonic API

| Commit | Feature | ADR | cuda-python Ref | CUDA History | Lines | Learning Focus |
|--------|---------|-----|-----------------|--------------|-------|----------------|
| 6.1 | High-level API | ADR-019 | cuda_core/experimental/__init__.py | Unified interface | 200 | API design |
| 6.2 | Example programs | ADR-020 | cuda_core/examples/ | Common patterns | 200 | Real-world usage |
| 6.3 | Documentation | ADR-021 | cuda_core/docs/ | User guide | 100 | Teaching others |

**Checkpoint 6**: Run vector add, matrix multiply, reduction examples

---

## 📝 Architecture Decision Record Template

```markdown
# ADR-XXX: [Decision Title]

**Status**: Accepted
**Date**: 2025-01-XX
**Commit**: [hash]
**cuda-python Reference**: [file:line]
**CUDA Version**: X.X (Year)

## Context

What problem are we solving?
What constraints exist?

## Decision

What approach did we choose?

### Code Example

```python
# Our mini-cuda implementation
[code snippet]
```

## Rationale

### Why This Approach?
1. Reason 1
2. Reason 2

### Alternatives Considered
- **Alternative A**: [why rejected]
- **Alternative B**: [why rejected]

## cuda-python Comparison

### What cuda-python Does
[Explanation + file references]

**File**: `/home/user/cuda-python/...`
**Key Functions**: `function_name()` (line XXX)

### What We're Doing Differently
[Simplifications + rationale]

## Historical Evolution

- **CUDA 1.0 (2007)**: [original approach]
- **CUDA 5.0 (2012)**: [major change]
- **CUDA 12.x (2024)**: [modern approach]

## Trade-offs

### Benefits
✅ Benefit 1
✅ Benefit 2

### Limitations
❌ Limitation 1
❌ Limitation 2

## Learning Outcomes

After this commit, you should understand:
1. [Concept 1]
2. [Concept 2]
3. [Concept 3]

## References

- **cuda-python**: `/home/user/cuda-python/[file]:[lines]`
- **CUDA Docs**: [URL]
- **Paper**: [if applicable]

## Exercises

1. [Hands-on exercise]
2. [Extension challenge]
3. [Debugging task]
```

---

## 🎯 Learning Checkpoint Template

```markdown
# Phase X Checkpoint: [Phase Name]

## Self-Assessment Quiz

### Conceptual Understanding

1. **Question**: Why do we use streams in CUDA?
   - **Answer**: To enable concurrent kernel execution and overlap computation with data transfers
   - **Reference**: ADR-009

2. [More questions...]

### Code Comprehension

1. **Question**: What does this code do?
   ```python
   buffer = DeviceBuffer(1024)
   buffer.copy_from_host(data, stream=s)
   ```
   - **Answer**: Allocates 1KB device memory and asynchronously copies host data using stream `s`

## Hands-On Exercises

### Exercise 1: Implement Async Memcpy
- **Task**: Modify `DeviceBuffer` to support multiple concurrent copies
- **Difficulty**: ⭐⭐☆☆☆
- **Estimated Time**: 30 minutes
- **Learning Goal**: Understand stream-based async operations

**Hints**:
- Use multiple streams
- Consider synchronization points
- Check cuda-python's implementation in `/home/user/cuda-python/cuda_core/cuda/core/experimental/_memory/`

### Exercise 2: Debug Memory Leak
- **Task**: Find and fix the memory leak in the provided code
- **Difficulty**: ⭐⭐⭐☆☆
- **Bug Description**: `DeviceBuffer` objects are not freed on exception

## Comparative Analysis

### Mini-CUDA vs cuda-python

| Aspect | Mini-CUDA | cuda-python | Why Different? |
|--------|-----------|-------------|----------------|
| Bindings | ctypes (200 lines) | Cython (20k lines) | Learning focus vs performance |
| Memory | Simple malloc/free | Memory pools, async allocator | Simplicity vs efficiency |
| Streams | Basic sync only | Callbacks, priorities | MVP vs production |

## Performance Benchmark

Run `make benchmark-phase-X` and compare results:

- **Expectation**: [what should happen]
- **Bottleneck**: [where is it slow]
- **cuda-python's Solution**: [how they optimize]

## Next Steps

Before moving to Phase X+1, ensure you can:

- [ ] Explain [concept] to someone else
- [ ] Modify the code to add [feature]
- [ ] Identify the performance bottleneck
- [ ] Read the equivalent cuda-python code comfortably
```

---

## 🔧 Project Structure

```
mini-cuda/
├── README.md                    # Project overview
├── LEARNING_GUIDE.md            # This document
├── HISTORICAL_TIMELINE.md       # CUDA evolution mapped to commits
│
├── docs/
│   ├── adrs/                    # Architecture Decision Records
│   │   ├── 001-bindings-layer-design.md
│   │   ├── 002-memory-management.md
│   │   └── ...
│   │
│   ├── comparisons/             # Mini vs cuda-python
│   │   ├── bindings-comparison.md
│   │   ├── memory-comparison.md
│   │   └── ...
│   │
│   ├── diagrams/                # Visual architecture
│   │   ├── memory-hierarchy.svg
│   │   ├── stream-pipeline.svg
│   │   └── ...
│   │
│   ├── checkpoints/             # Learning checkpoints
│   │   ├── phase1-checkpoint.md
│   │   ├── phase2-checkpoint.md
│   │   └── ...
│   │
│   └── references/              # External references
│       ├── cuda-docs.md
│       ├── cuda-python-refs.md
│       └── papers.md
│
├── mini_cuda/
│   ├── __init__.py
│   │
│   ├── bindings/                # Phase 1
│   │   ├── __init__.py
│   │   ├── runtime.py           # CUDA Runtime API
│   │   ├── driver.py            # CUDA Driver API (minimal)
│   │   ├── types.py             # CUDA types & enums
│   │   ├── errors.py            # Error handling
│   │   └── loader.py            # Library loading
│   │
│   ├── core/                    # Phase 2-5
│   │   ├── __init__.py
│   │   ├── device.py            # Device management
│   │   ├── memory.py            # Memory management
│   │   ├── stream.py            # Stream management
│   │   ├── event.py             # Event management
│   │   ├── kernel.py            # Kernel launch
│   │   └── module.py            # Module loading
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Utility functions
│
├── tests/
│   ├── unit/                    # Unit tests
│   │   ├── test_bindings.py
│   │   ├── test_device.py
│   │   ├── test_memory.py
│   │   └── ...
│   │
│   ├── integration/             # Integration tests
│   │   ├── test_vector_add.py
│   │   ├── test_matrix_mul.py
│   │   └── ...
│   │
│   ├── benchmarks/              # Performance tests
│   │   ├── bench_memcpy.py
│   │   ├── bench_kernel_launch.py
│   │   └── ...
│   │
│   └── exercises/               # Learning exercises
│       ├── exercise1_async_copy.py
│       ├── exercise2_multi_stream.py
│       └── ...
│
├── examples/
│   ├── 01_device_query.py       # List GPU properties
│   ├── 02_memory_copy.py        # Basic H2D/D2H copy
│   ├── 03_vector_add.py         # Vector addition kernel
│   ├── 04_async_streams.py      # Stream-based async
│   ├── 05_event_timing.py       # Event-based timing
│   └── 06_matrix_multiply.py    # 2D grid/block
│
├── kernels/                     # Pre-compiled PTX/CUBIN
│   ├── vector_add.ptx
│   ├── matrix_mul.ptx
│   └── reduction.ptx
│
├── tools/
│   ├── ptx_compiler/            # Compile CUDA to PTX
│   ├── visualizer/              # Visualize kernel launch
│   └── profiler/                # Performance profiling
│
├── Makefile                     # Build & test commands
├── setup.py                     # Package installation
├── requirements.txt             # Dependencies
└── pytest.ini                   # Test configuration
```

---

## 🎓 Commit Message Format

```
[Phase X.Y] Title - Historical Context

Brief description of what this commit implements.

cuda-python Reference: cuda_core/experimental/_module.py:100-200
CUDA Version: X.X (Year)
Historical Note: This mirrors CUDA X.X's feature introduction

Design Decisions:
- Decision 1: [rationale]
- Decision 2: [rationale]

Trade-offs:
- We simplified [X] because [Y]
- We kept [A] to preserve [B]

Learning Outcomes:
1. Understand [concept]
2. See how [feature] works
3. Compare [approach A] vs [approach B]

See docs/adrs/ADR-XXX.md for full decision record.
```

---

## 📊 What's New in This Approach?

| Enhancement | Benefit |
|-------------|---------|
| **Architecture Decision Records** | Understand *why* not just *what* |
| **Historical Timeline** | See CUDA's evolution mapped to our code |
| **Learning Checkpoints** | Active learning, not passive reading |
| **Comparative Analysis** | Explicit mini vs cuda-python differences |
| **cuda-python References** | Learn from production code |
| **CUDA Docs Integration** | Understand official CUDA concepts |
| **Hands-on Exercises** | Apply knowledge, extend the code |
| **Performance Benchmarks** | Understand optimization trade-offs |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x or 12.x
- `nvidia-cuda-runtime-cu12` or similar

### Installation

```bash
# Clone the repo (or create mini-cuda folder)
cd /home/user/cuda-python
mkdir mini-cuda
cd mini-cuda

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pytest

# Start with Phase 1, Commit 1.1
```

### First Steps

1. **Study cuda-python**: Read `/home/user/cuda-python/cuda_bindings/README.md`
2. **Read ADR-001**: Understand binding layer design decisions
3. **Implement Phase 1.1**: Library loading with ctypes
4. **Run tests**: `pytest tests/unit/test_bindings.py`
5. **Complete Checkpoint 1**: Self-assessment quiz
6. **Move to Phase 1.2**: Device & memory APIs

---

## 🎯 Learning Goals

By the end of this project, you should:

### Understand GPU Programming
- [ ] CUDA execution model (grids, blocks, threads)
- [ ] Memory hierarchy (global, shared, local)
- [ ] Asynchronous execution (streams, events)
- [ ] Kernel launch mechanics

### Understand Python-CUDA Integration
- [ ] ctypes/cffi for C library binding
- [ ] Memory management (host/device)
- [ ] Numpy integration
- [ ] Error handling across language boundary

### Read Production Code
- [ ] Navigate cuda-python codebase
- [ ] Understand Cython optimizations
- [ ] See real-world API design patterns
- [ ] Learn from NVIDIA's implementations

### Design APIs
- [ ] Pythonic vs low-level tradeoffs
- [ ] RAII patterns in Python
- [ ] Context managers for resources
- [ ] Error propagation strategies

---

## 📚 References

### cuda-python Source Code

#### Bindings Layer
- `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/utils.pyx`
- `/home/user/cuda-python/cuda_bindings/cuda/bindings/_internal/nvvm_linux.pyx`

#### Core Layer
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_device.pyx`
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_stream.pyx`
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_event.pyx`
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_memory/`
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_module.py`
- `/home/user/cuda-python/cuda_core/cuda/core/experimental/_launcher.pyx`

#### Examples
- `/home/user/cuda-python/cuda_core/examples/show_device_properties.py`
- `/home/user/cuda-python/cuda_bindings/examples/`

### CUDA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/)

### Academic Papers
- "CUDA Programming Model" (NVIDIA, 2007)
- "Efficient Sparse Matrix-Vector Multiplication on CUDA" (Bell & Garland, 2008)
- "Streaming Memory Hierarchy" (Rogers et al., 2009)

---

## ✅ Next Steps

1. **Create Project Structure**: Set up folders and files
2. **Write ADR-001**: Bindings layer design decisions
3. **Implement Phase 1.1**: Library loading with ctypes
4. **Write Tests**: Basic binding tests
5. **Create Checkpoint 1**: Self-assessment quiz

Ready to start building? Let's go! 🚀

---

## 📝 Notes

- **Focus on learning, not production code**: Simplify aggressively
- **Test often**: Each commit should have passing tests
- **Document decisions**: ADRs are for future you
- **Compare frequently**: Check cuda-python to validate approach
- **Ask "why?"**: Don't just copy, understand the reasoning

Good luck on your CUDA learning journey! 🎓
