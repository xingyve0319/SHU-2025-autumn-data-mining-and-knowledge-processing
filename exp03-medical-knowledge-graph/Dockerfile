# 1. 选择基础镜像 (推荐 python slim 版本，体积小)
FROM python:3.10-slim-bookworm

# 2. 【核心魔法】直接从 uv 官方镜像里把 uv 的二进制文件“偷”过来
# 这比 pip install uv 要快得多，而且不需要装 curl/wget
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. 设置环境变量
# UV_COMPILE_BYTECODE=1: 安装时编译 .pyc 文件，加快容器启动速度
# UV_LINK_MODE=copy: 在 Docker 里不要用硬链接，直接复制文件 (防止跨层问题)
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 4. 设置工作目录
WORKDIR /app

# 5. 【关键优化】先只复制配置文件！
# 这样如果你的代码变了但依赖没变，Docker 会直接使用缓存，跳过安装步骤
COPY pyproject.toml uv.lock ./

# 6. 安装依赖
# --frozen: 严格按照 uv.lock 安装 (类似 npm ci)，确保环境绝对一致
# --no-install-project: 只装依赖库，先不装你自己的代码 (为了利用缓存)
RUN uv sync --frozen --no-install-project

# 7. 此时再复制剩下的代码
COPY . .

# 8. 安装你自己的项目 (如果有) 
# 如果你的项目本身不是一个包，这一步其实可以省略，或者用 --no-dev
RUN uv sync --frozen

# 9. 【重要】把 uv 创建的虚拟环境加入 PATH
# 这样后面的命令 (python, torch) 都会自动使用 .venv 里的
ENV PATH="/app/.venv/bin:$PATH"

# 10. 启动命令
CMD ["python", "main.py"]