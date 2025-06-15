FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /app
COPY . .

RUN uv python install 3.10 && \
    uv venv --prompt medAlapaca && \
    . .venv/bin/activate && \
    uv pip install -r /app/requirements.txt