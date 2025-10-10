# syntax=docker/dockerfile:1.7

# 使用与项目一致的 Python 版本
FROM python:3.13.3-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_ENV=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 安装基础工具（用于安装 uv 以及少量构建依赖）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（更快的包管理器）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

# 为外部卷预创建挂载点（镜像不包含实际数据）
RUN mkdir -p /app/memory /app/configs

# 复制项目源码（受 .dockerignore 控制，避免把 memory/ 和 configs/ 打进镜像）
COPY . .

# 使用 pyproject 安装项目及依赖（使用系统环境安装）
RUN uv pip install --system --no-cache .

# 暴露服务端口
EXPOSE 8000

# 运行时环境（可在 docker-compose 或 -e 覆盖）
ENV HOST=0.0.0.0 \
    PORT=8000

# 入口：直接启动 Web API
ENTRYPOINT ["python", "src/MiraMate/web_api/start_web_api.py"]
