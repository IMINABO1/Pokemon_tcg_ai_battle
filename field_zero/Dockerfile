# RHEL-compatible base with CUDA preinstalled, matching Quest's driver family.
# nvidia publishes official Rocky Linux 9 images with the CUDA toolkit baked in.
# "runtime" (not "devel"): we only ever run precompiled PyTorch kernels here,
# never compile CUDA C++ ourselves, so the compiler toolchain (nvcc) and
# static libs in "devel" are dead weight -- this alone cuts several GB.
FROM nvidia/cuda:12.4.1-cudnn-runtime-rockylinux9

# Rocky/RHEL ships Python 3.9 by default; get 3.11 from the module stream.
RUN dnf install -y python3.11 python3.11-pip git && \
    dnf clean all && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
# CUDA 12.4 wheel index — must match the CUDA version in the base image tag above.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /workspace/field_zero

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/field_zero
