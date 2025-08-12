#\!/usr/bin/env python
"""
Test script for LFMC model with CUDA extensions
"""

import os
import sys
import torch

# Set CUDA environment
os.environ['CUDA_HOME'] = '/packages/apps/spack/18/opt/spack/gcc-11.2.0/cuda-11.7.0-bhi'
os.environ['PATH'] = os.environ['CUDA_HOME'] + '/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = os.environ['CUDA_HOME'] + '/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

print("=" * 60)
print("LFMC Model with CUDA Extensions Test")
print("=" * 60)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

print(f"\nEnvironment:")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")

# Try importing the model
print("\nImporting LFMC model...")
sys.path.insert(0, '/scratch/qhuang62/Grid4D')
sys.path.insert(0, '/scratch/qhuang62/Grid4D/LFMC')

try:
    from lfmc_model import LFMCPredictor
    print("✓ Successfully imported LFMCPredictor\!")
    
    # Test model creation
    print("\nCreating model...")
    model = LFMCPredictor(
        grid4d_config={
            'canonical_num_levels': 16,
            'canonical_level_dim': 2,
            'canonical_base_resolution': 16,
            'canonical_desired_resolution': 256,
            'canonical_log2_hashmap_size': 19,
            'deform_num_levels': 16,
            'deform_level_dim': 2,
            'deform_base_resolution': [8, 8, 8],
            'deform_desired_resolution': [32, 32, 16],
            'deform_log2_hashmap_size': 19,
            'bound': 1.0
        },
        hidden_dims=[256, 128, 64],
        use_attention=True,
        dropout_rate=0.1
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model moved to CUDA\!")
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_input = torch.randn(32, 4)  # batch_size=32, (x,y,z,t)
    if torch.cuda.is_available():
        test_input = test_input.cuda()
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✓ Forward pass successful\!")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Print model info
    param_info = model.get_num_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {param_info['total']:,}")
    print(f"  Trainable: {param_info['trainable']:,}")
    print(f"  Encoder: {param_info['encoder']:,}")
    print(f"  Attention: {param_info['attention']:,}")
    print(f"  MLP: {param_info['mlp']:,}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Model is ready for training\!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error importing or testing model:")
    print(f"  {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nPlease check your CUDA installation and environment variables.")
    sys.exit(1)
