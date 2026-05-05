# Reproducibility — NeurIPS 2026 supervised pipeline

## Git state
- Commit SHA recorded in `repro_git_sha.txt` after each `reproduce.sh` run

## System
- OS: Darwin 25.3.0 (arm64)
- Python: 3.13.12 (main, Feb 12 2026, 01:06:02) [Clang 21.1.4 ]
- numpy.show_config:
  ```
  Build Dependencies:
    blas:
      detection method: system
      found: true
      include directory: unknown
      lib directory: unknown
      name: accelerate
      openblas configuration: unknown
      pc file directory: unknown
      version: unknown
    lapack:
      detection method: internal
      found: true
      include directory: unknown
      lib directory: unknown
      name: dep4368281552
      openblas configuration: unknown
      pc file directory: unknown
      version: 1.26.4
  Compilers:
    c:
      commands: cc
      linker: ld64
      name: clang
      version: 17.0.0
    c++:
      commands: c++
      linker: ld64
      name: clang
      version: 17.0.0
    cython:
      commands: cython
      linker: cython
      name: cython
      version: 3.0.12
  Machine Information:
    build:
      cpu: aarch64
      endian: little
      family: aarch64
      system: darwin
    host:
      cpu: aarch64
      endian: little
      family: aarch64
      system: darwin
  Python Information:
    version: '3.13'
  SIMD Extensions:
    baseline:
    - NEON
    - NEON_FP16
    - NEON_VFPV4
    - ASIMD
    found:
    - ASIMDHP
    not found:
    - ASIMDFHM
  
  
  ```

## Hardware (inferred)
- MPS available: True
- CUDA available: False

## Pipeline-specific notes
- BLAS backend differences: ~1e-6 drift in PCA; pairwise accuracy
  unaffected at 3-decimal precision.
- See input_cache_sha256.txt for the canonical input-artifact hashes;
  any divergence from those is a provenance red flag.
