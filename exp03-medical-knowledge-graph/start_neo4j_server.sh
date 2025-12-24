#!/bin/bash
echo "🚀 正在启动 Neo4j 容器..."

# 确保目录存在
mkdir -p data/neo4j/data
mkdir -p data/neo4j/logs
mkdir -p data/neo4j/import
mkdir -p data/neo4j/plugins

# 如果容器已经存在，先删除
if [ "$(docker ps -aq -f name=neo4j)" ]; then
    echo "发现旧容器，正在清理..."
    docker rm -f neo4j
fi

# 启动新容器 (纯净命令，无干扰注释)
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -v $(pwd)/data/neo4j/data:/data \
    -v $(pwd)/data/neo4j/logs:/logs \
    -v $(pwd)/data/neo4j/import:/var/lib/neo4j/import \
    -v $(pwd)/data/neo4j/plugins:/plugins \
    -e NEO4J_AUTH=neo4j/12345678 \
    m.daocloud.io/docker.io/neo4j:latest

echo "✅ Neo4j 启动成功！"
echo "⏳ 请等待 15 秒让数据库初始化，然后访问 http://localhost:7474"
