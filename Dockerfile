FROM vemlp-cn-beijing.cr.volces.com/preset-images/python:3.12-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace/Auxiliary-loss-free


COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["sleep", "infinity"]