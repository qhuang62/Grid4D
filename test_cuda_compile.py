"""
Test script to diagnose and fix CUDA compilation issues
"""

import os
import sys
import torch
from pathlib import Path

print("=" * 60)
print("CUDA Compilation Diagnostic")
print("=" * 60)

# Check PyTorch and CUDA info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Try to find CUDA paths
from torch.utils.cpp_extension import CUDA_HOME, _find_cuda_home

print(f"\nCUDA_HOME from env: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"PyTorch CUDA_HOME: {CUDA_HOME}")

# Try to find nvcc
try:
    # Set CUDA_HOME based on PyTorch's CUDA
    if CUDA_HOME is None:
        # Try common locations
        cuda_paths = [
            '/usr/local/cuda',
            '/usr/local/cuda-12.6',
            '/usr/local/cuda-12.5',
            '/usr/local/cuda-12.4',
            '/usr/local/cuda-12.3',
            '/usr/local/cuda-12.2',
            '/usr/local/cuda-12.1',
            '/usr/local/cuda-12.0',
            '/usr/local/cuda-11.8',
            '/usr/local/cuda-11.7',
        ]
        
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc')
                if os.path.exists(nvcc_path):
                    os.environ['CUDA_HOME'] = cuda_path
                    print(f"\nFound CUDA at: {cuda_path}")
                    print(f"Setting CUDA_HOME to: {cuda_path}")
                    break
        else:
            # Try to use PyTorch's bundled CUDA
            print("\nNo system CUDA found, attempting to use PyTorch's bundled CUDA...")
            
            # Get PyTorch's include and library paths
            torch_dir = os.path.dirname(torch.__file__)
            include_dirs = torch.utils.cpp_extension.include_paths()
            library_dirs = torch.utils.cpp_extension.library_paths()
            
            print(f"PyTorch include dirs: {include_dirs}")
            print(f"PyTorch library dirs: {library_dirs}")
            
            # Create a fake CUDA_HOME structure
            fake_cuda_home = Path('./cuda_home_fake')
            fake_cuda_home.mkdir(exist_ok=True)
            (fake_cuda_home / 'bin').mkdir(exist_ok=True)
            (fake_cuda_home / 'include').mkdir(exist_ok=True)
            (fake_cuda_home / 'lib64').mkdir(exist_ok=True)
            
            # Create a dummy nvcc script that uses nvcc from conda if available
            nvcc_script = fake_cuda_home / 'bin' / 'nvcc'
            nvcc_script.write_text('''#!/bin/bash
# Dummy nvcc script that uses PyTorch's CUDA compilation
echo "Using PyTorch's CUDA compilation wrapper"
python -m torch.utils.cpp_extension _nvcc "$@"
''')
            nvcc_script.chmod(0o755)
            
            os.environ['CUDA_HOME'] = str(fake_cuda_home.absolute())
            print(f"Created fake CUDA_HOME at: {fake_cuda_home.absolute()}")
    
    # Try a simple compilation test
    print("\n" + "=" * 60)
    print("Testing simple CUDA compilation...")
    print("=" * 60)
    
    from torch.utils.cpp_extension import load_inline
    
    cuda_source = '''
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    
    __global__ void test_kernel(float* data, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = data[idx] * 2.0f;
        }
    }
    
    torch::Tensor test_cuda_function(torch::Tensor input) {
        auto output = input.clone();
        int size = output.numel();
        
        AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "test_cuda_function", ([&] {
            test_kernel<<<(size + 255) / 256, 256>>>(
                output.data_ptr<scalar_t>(),
                size
            );
        }));
        
        return output;
    }
    '''
    
    cpp_source = '''
    #include <torch/extension.h>
    torch::Tensor test_cuda_function(torch::Tensor input);
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("test_cuda_function", &test_cuda_function, "Test CUDA function");
    }
    '''
    
    try:
        # Create a build directory
        build_dir = Path('./test_cuda_build')
        build_dir.mkdir(exist_ok=True)
        
        print("Attempting to compile test CUDA extension...")
        test_module = load_inline(
            name='test_cuda',
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            build_directory=str(build_dir),
            verbose=False,
            extra_cuda_cflags=['-O2']
        )
        
        # Test the compiled module
        test_tensor = torch.randn(10, device='cuda')
        result = test_module.test_cuda_function(test_tensor)
        expected = test_tensor * 2
        
        if torch.allclose(result, expected):
            print("✓ CUDA compilation test PASSED!")
            print("The system can compile CUDA extensions.")
        else:
            print("✗ CUDA compilation test FAILED!")
            print("Module compiled but gave incorrect results.")
            
    except Exception as e:
        print(f"✗ CUDA compilation test FAILED with error:")
        print(f"  {str(e)}")
        print("\nThis means we cannot compile custom CUDA extensions.")
        print("We'll need to use the simplified model without custom CUDA kernels.")
        
except Exception as e:
    print(f"\nError during diagnostic: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)