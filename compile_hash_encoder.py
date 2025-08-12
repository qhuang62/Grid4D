"""
Compile hash encoder extension with proper settings
"""

import os
import sys
import torch
from torch.utils.cpp_extension import load

# Set environment
os.environ['CUDA_HOME'] = '/packages/apps/spack/18/opt/spack/gcc-11.2.0/cuda-11.7.0-bhi'
os.environ['PATH'] = os.environ['CUDA_HOME'] + '/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = os.environ['CUDA_HOME'] + '/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Force C++14 standard to avoid compatibility issues
os.environ['CXXFLAGS'] = '-std=c++14'

print("Compiling hash encoder extension...")
print(f"CUDA_HOME: {os.environ['CUDA_HOME']}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")

_src_path = os.path.dirname(os.path.abspath(__file__)) + '/hashencoder'

try:
    # Clean build directory
    import shutil
    if os.path.exists('./tmp_build'):
        shutil.rmtree('./tmp_build')
    
    _backend = load(
        name='_hash_encoder',
        extra_cflags=['-O3', '-std=c++14'],
        extra_cuda_cflags=[
            '-O3', '-std=c++14', 
            '-allow-unsupported-compiler',
            '-U__CUDA_NO_HALF_OPERATORS__', 
            '-U__CUDA_NO_HALF_CONVERSIONS__', 
            '-U__CUDA_NO_HALF2_OPERATORS__',
            '-gencode', 'arch=compute_80,code=sm_80',  # A100 architecture
        ],
        sources=[
            os.path.join(_src_path, 'src', 'hashencoder.cu'),
            os.path.join(_src_path, 'src', 'bindings.cpp'),
        ],
        build_directory='./tmp_build/',
        verbose=True,
        with_cuda=True
    )
    
    print("\n✓ Hash encoder compiled successfully!")
    print(f"Module location: {_backend}")
    
except Exception as e:
    print(f"\n✗ Failed to compile hash encoder:")
    print(str(e))
    sys.exit(1)