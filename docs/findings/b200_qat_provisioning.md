# B200 provisioned for prospective quantization-aware training

At the user's request, one B200 was provisioned on 2026-10-12 as Pod
`tv6i08lhsf2cfy` (`gleipnir-b200-qat`) in US-CA-2 at $7.99/hour.
Live catalog reads showed no B200 capacity in the preferred EU-RO-1 or
previous US-NC-2 region; US-CA-2 was the available fallback.

Direct SSH and a finite BF16 matrix multiplication pass. The GPU is
`GPU-3548f8c1-43e7-672c-b2e9-29df10fefcb5`, with 183,359 MiB VRAM,
compute capability 10.0 and driver 595.91.07. The image is
`runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`; its initial
Python 3.12 runtime has Torch 2.9.1+cu130/CUDA 13.0 and lacks NumPy.
This verifies hardware operation, not the locked project training environment.
No QAT experiment or training process has been launched.

The Pod has a 100 GB Pod-attached `/workspace` and a 50 GB container disk.
Published running storage rates imply $10/month and $5/month respectively,
in addition to compute. The workspace survives stopping but is deleted on
termination; collect important artifacts first. No independent network volume
was created, and existing EU/NC2 network volumes were not attached or modified.

Sanitized allocation metadata is saved locally under
`results/b200_provisioning/20261012_qat/`.

The user subsequently requested termination to restrict future capacity to
US-NC-2 or EU regions with existing setup. The delete returned HTTP 204 and
an independent lookup returned HTTP 404 on 2026-10-12. Termination deletes
the Pod-attached workspace and container disk; no independent CA2 volume
exists. Hardware receipts were already collected locally. The connection
alias is marked TERMINATED, and no replacement Pod was launched. A live
network-volume inventory confirms retained EU volume `cb7bxu71ug` and NC2
volume `ixbh81vf9c`. The termination receipt is beside the allocation metadata.
