# CUDA Microkernels and Parallel Algorithms (C++)

A collection of independent CUDA kernels and small C++ programs
demonstrating GPU parallel programming patterns, memory hierarchy usage,
and performance-oriented design.

This repository contains standalone CUDA examples focused on:

- Parallel reduction (block-level, warp-level)
- Dot product and vector operations
- Memory access patterns and coalescing
- Shared memory usage and synchronization
- Atomic operations and their trade-offs
- Multi-stage kernel design

Each example is intentionally isolated to highlight a specific GPU concept
or performance pattern rather than a full application.

## Example: Parallel Reduction

The reduction examples demonstrate different approaches to summation:

- Block-level reduction using shared memory
- Warp-level reduction using warp primitives
- Atomic-based reduction for simplicity and comparison

The goal is to expose the trade-offs between simplicity, synchronization
overhead, and scalability.
