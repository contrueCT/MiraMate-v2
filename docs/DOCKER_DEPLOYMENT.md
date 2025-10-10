# 使用 Docker 构建与部署 MiraMate

本文档介绍如何在 Windows 与 Linux 环境下，从项目构建 Docker 镜像并运行容器，包括使用 docker compose 与单条 docker 命令的两套方式。

## 前提条件

- 已安装 Docker（Docker Desktop for Windows 或 Linux 上的 Docker Engine）
- 建议安装 Compose V2（`docker compose` 命令）。
- 在项目根目录执行以下命令（根目录包含 `Dockerfile` 和 `docker-compose.yml`）。

## 关键约定与目录映射

- 容器工作目录：`/app`
- 外部卷挂载（数据与配置持久化）：
  - 宿主 `./configs` 映射为容器 `/app/configs`
  - 宿主 `./memory` 映射为容器 `/app/memory`
- 入口：`python src/MiraMate/web_api/start_web_api.py`
- 端口：默认 `8000`（可通过环境变量覆盖）
- 环境变量：
  - `DOCKER_ENV=1`（容器模式标识）
  - `HOST=0.0.0.0`（在容器内监听所有地址）
  - `PORT=8000`
- 主要访问地址：
  - 健康检查: `http://localhost:8000/api/health`
  - OpenAPI 文档: `http://localhost:8000/docs`

> 提示：仓库中的 `.dockerignore` 已排除 `memory/` 与部分 `configs` 下的敏感文件，避免它们被打包进镜像。容器运行时通过卷挂载使用主机上的数据与配置。

---

## 方式一：使用 docker compose（推荐）

### Windows（PowerShell）

- 构建镜像：

```powershell
docker compose build
```

- 后台运行：

```powershell
docker compose up -d
```

- 查看日志：

```powershell
docker compose logs -f
```

- 停止并移除容器：

```powershell
docker compose down
```

### Linux（bash）

- 构建镜像：

```bash
docker compose build
```

- 后台运行：

```bash
docker compose up -d
```

- 查看日志：

```bash
docker compose logs -f
```

- 停止并移除容器：

```bash
docker compose down
```

> 说明：`docker-compose.yml` 已包含端口映射（`8000:8000`）、环境变量（`DOCKER_ENV/HOST/PORT`）与卷挂载（`./configs:/app/configs`、`./memory:/app/memory`），开箱即用。

---

## 方式二：使用单条 docker 命令

### 1) 构建镜像

- Windows / Linux 通用（在项目根目录）：

```bash
docker build -t miramate-api:latest .
```

### 2) 运行容器

- Windows（PowerShell）：

```powershell
docker run -d --name miramate-api `
  -p 8000:8000 `
  -e DOCKER_ENV=1 -e HOST=0.0.0.0 -e PORT=8000 `
  -v "${PWD}\configs:/app/configs" `
  -v "${PWD}\memory:/app/memory" `
  miramate-api:latest
```

- Linux（bash）：

```bash
docker run -d --name miramate-api \
  -p 8000:8000 \
  -e DOCKER_ENV=1 -e HOST=0.0.0.0 -e PORT=8000 \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/memory:/app/memory" \
  miramate-api:latest
```

- 查看日志：

```bash
docker logs -f miramate-api
```

- 停止并移除容器：

```bash
docker stop miramate-api && docker rm miramate-api
```

---

## 访问接口

- 健康检查: `http://localhost:8000/api/health`
- API 文档: `http://localhost:8000/docs`
- WebSocket: `ws://localhost:8000/ws`

> 首次运行若 `./configs/llm_config.json` 未配置有效 API Key，服务仍会启动，但部分功能需在配置后重启生效。

---

## 常见问题（FAQ）

1. 端口被占用（`bind: address already in use`）

- 修改映射端口，例如 `-p 18000:8000` 或在 `docker-compose.yml` 中改为 `"18000:8000"`。

2. Windows 卷挂载失败

- 请先在项目根目录创建 `configs` 与 `memory` 文件夹；确保 Docker Desktop 已开启对应驱动器的文件共享权限。

3. Linux 卷权限问题

- 确保当前用户在 `docker` 组内或使用 `sudo`；目录权限允许容器读写：`chmod -R u+rwX ./configs ./memory`。
- SELinux 环境（如 CentOS/RHEL/Fedora）可在挂载后添加 `:Z` 标志：
  - `-v "$(pwd)/configs:/app/configs:Z" -v "$(pwd)/memory:/app/memory:Z"`

4. 构建缓慢或失败

- 构建需要网络访问 PyPI/Hugging Face 等源；若在内网或代理环境，请配置镜像加速或代理。
- 本项目包含 `sentence-transformers`（依赖较大），首次构建时间较长属正常。

5. 升级/回滚

- 使用 compose：
  - 重新构建：`docker compose build --no-cache`
  - 重启：`docker compose up -d`
- 使用镜像仓库分发时：先 `docker pull` 新版本，再 `docker compose up -d` 或 `docker run` 重新启动。

---

## 可选：发布镜像到镜像仓库

- 标记与推送到 Docker Hub：

```bash
docker login

docker tag miramate-api:latest <YOUR_DOCKERHUB_USERNAME>/miramate-api:latest

docker push <YOUR_DOCKERHUB_USERNAME>/miramate-api:latest
```

- 目标机器拉取并运行：

```bash
docker pull <YOUR_DOCKERHUB_USERNAME>/miramate-api:latest

docker run -d --name miramate-api \
  -p 8000:8000 \
  -e DOCKER_ENV=1 -e HOST=0.0.0.0 -e PORT=8000 \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/memory:/app/memory" \
  <YOUR_DOCKERHUB_USERNAME>/miramate-api:latest
```

---

## 目录与文件说明

- `Dockerfile`：基于 `python:3.13.3-slim`，使用 `uv pip install .` 从 `pyproject.toml` 安装依赖与项目；设置 `DOCKER_ENV=1`；入口为 `start_web_api.py`。
- `.dockerignore`：与 `.gitignore` 对齐，排除 `memory/` 和敏感配置文件，缩小上下文与保护密钥。
- `docker-compose.yml`：定义服务、端口、环境变量与卷挂载，并包含健康检查。

如需定制（端口、环境变量、卷路径或加入反向代理/Nginx 等），可在 `docker-compose.yml` 中按需修改或新增服务。
