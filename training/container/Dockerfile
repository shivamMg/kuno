# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir --upgrade pip==25.0.1
RUN python -m pip install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir \
    transformers==5.16.1 accelerate==1.14.0 \
    tokenizers==0.23.1 safetensors==0.8.0 \
    pydantic==2.12.5 PyYAML==6.0.3 numpy==2.4.2
WORKDIR /opt/kuno
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/kuno /opt/kuno
COPY training/container/entrypoint.sh /usr/local/bin/kuno-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/kuno-entrypoint \
    && chmod 0755 /usr/local/bin/kuno-entrypoint \
    && useradd --create-home --shell /bin/bash kuno
USER kuno
WORKDIR /opt/kuno
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/kuno-entrypoint"]
