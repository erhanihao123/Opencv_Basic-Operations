#!/bin/bash
set -e

echo "=========================================="
echo "  图像标注工具 - 阿里云一键部署脚本"
echo "=========================================="

APP_NAME="image-annotation-tool"
REPO_URL="https://github.com/erhanihao123/Opencv_Basic-Operations.git"
PROJECT_DIR="/opt/$APP_NAME"

echo ""
echo "🚀 步骤1: 更新系统并安装依赖..."
apt update -y && apt upgrade -y
apt install -y git curl wget

echo ""
echo "🚀 步骤2: 安装 Docker..."
curl -fsSL https://get.docker.com | bash
systemctl start docker
systemctl enable docker

echo ""
echo "🚀 步骤3: 安装 Docker Compose..."
apt install -y docker-compose-plugin

echo ""
echo "🚀 步骤4: 克隆项目代码..."
rm -rf "$PROJECT_DIR"
git clone "$REPO_URL" "$PROJECT_DIR"

echo ""
echo "🚀 步骤5: 进入项目目录..."
cd "$PROJECT_DIR/web_opencv/deploy"

echo ""
echo "🚀 步骤6: 启动服务..."
docker-compose up -d

echo ""
echo "⏳ 等待服务启动...（约30秒）"
sleep 30

echo ""
echo "✅ 部署完成！"
echo ""
echo "=========================================="
echo "  访问地址: http://$(curl -s ifconfig.me)"
echo "  健康检查: http://$(curl -s ifconfig.me)/health"
echo "=========================================="
echo ""
echo "查看日志: docker-compose logs -f"
echo "停止服务: docker-compose down"