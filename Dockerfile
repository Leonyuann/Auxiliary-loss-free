FROM vemlp-cn-beijing.cr.volces.com/preset-images/python:3.12-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV NVTE_FRAMEWORK=pytorch \
    NVTE_BUILD_USE_NVIDIA_WHEELS=1 \
    NVTE_CUDA_ARCHS=80 \
    MAX_JOBS=4 \
    NVTE_BUILD_THREADS_PER_JOB=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace/Auxiliary-loss-free

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --group te-build \
        --no-install-package transformer-engine
RUN uv sync --frozen --no-install-project --group te-build

COPY . .
RUN uv sync --frozen --group te-build
RUN uv run --frozen --group te-build python -c "import transformer_engine.pytorch"

CMD ["sleep", "infinity"]
