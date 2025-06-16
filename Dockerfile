FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04
# CUDA setup
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# medAlpaca setup
WORKDIR /app
COPY . .
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv python install 3.10.8 && \
    uv venv --prompt medAlapaca && \
    . .venv/bin/activate && \
    uv pip install -r /app/requirements.txt
# Container setup
RUN echo '#!/bin/bash\nset -e\nsource /app/.venv/bin/activate\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]