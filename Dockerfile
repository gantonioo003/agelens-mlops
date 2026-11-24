FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

COPY --from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /workspace
RUN useradd --create-home --uid 10001 agelens
COPY requirements-runtime.lock /tmp/requirements-runtime.lock
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --link-mode copy -r /tmp/requirements-runtime.lock
COPY --chown=agelens:agelens configs ./configs
COPY --chown=agelens:agelens scripts ./scripts
RUN mkdir -p /state && chown agelens:agelens /state
USER agelens
CMD ["python", "-m", "scripts.check_environment", "--minimal"]
