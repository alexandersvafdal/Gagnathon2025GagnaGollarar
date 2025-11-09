FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DASH_DEBUG_MODE=False

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

WORKDIR /app

RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . .
EXPOSE 8050
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app.app:server"]

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

USER nonroot

CMD ["gunicorn", "-b", "0.0.0.0:8050", "app.app:server"]