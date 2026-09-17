# Formula Fly training/runtime image.
#
# Base: NVIDIA NGC PyTorch (multi-arch where NVIDIA publishes it).
# Default tag is the generic py3 image. On DGX Spark (aarch64) override:
#
#   podman build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3 .
#
# Confirm the tag exists for linux/arm64 before using it on Spark.
# This file has not been built on a Spark in this change; CI runs pytest on
# GitHub-hosted x86 runners without this image.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/formula-fly

# Copy lockfiles first so dependency layers cache.
COPY requirements-lock.txt requirements-aarch64.txt pyproject.toml README.md LICENSE ./
COPY fly_driver ./fly_driver
COPY tests ./tests

# NGC images already provide torch. Install only the project and CI extras.
# flyvis / gymnasium / wandb stay optional: add with `--extra eye` / `train` / `logging`.
RUN python -m pip install --no-cache-dir -r requirements-lock.txt \
    && python -m pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "-q"]
