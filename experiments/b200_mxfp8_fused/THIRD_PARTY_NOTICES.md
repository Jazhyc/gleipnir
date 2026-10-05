# Source and licenses

The fused producer is a project implementation of the single-pass consumer
scale-layout and transpose-invariant quantization ideas described in the Meta
blog. No unpublished Meta RMSNorm/GEMM implementation is copied.

Scale addressing and the E8M0 PTX conversion follow NVIDIA cuDNN Frontend at
revision `51d9d06b574222378a3d806009accab098e73705` and the existing project
packed MXFP8 port. NVIDIA dependency/derivative notices and license texts are
retained in [`b200_nvidia_mxfp8_varlen`](../b200_nvidia_mxfp8_varlen/THIRD_PARTY_NOTICES.md).
The existing native attention kernels remain isolated upstream dependencies.
