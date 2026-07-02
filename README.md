# Opencv_Basic-Operations
基于 Streamlit 和 OpenCV 的 Web 端图像标注工具，支持边缘检测、人脸检测、目标检测以及丰富的图像处理功能
 Web端图像标注工具

基于 Streamlit 和 OpenCV 的 Web 端图像标注工具，支持边缘检测、人脸检测、目标检测以及丰富的图像处理功能。

## 📋 功能特性

### 📊 检测模块
- **边缘检测**：使用 Canny 算法进行边缘检测，支持阈值参数调节
- **人脸检测**：使用 Haar 级联分类器检测图像中的人脸，支持多分类器融合、眼睛检测、肤色过滤
- **目标检测**：使用 MobileNet-SSD 模型进行目标检测，支持 20 种常见物体

### 🛠️ 基础图像处理
- **色彩空间转换**：支持灰度、HSV、RGB 色彩空间转换
- **几何变换**：支持缩放、旋转、平移、仿射变换
- **图像阈值化**：支持二值化、反二值化、截断、归零、自适应阈值、Otsu 自动阈值
- **平滑/滤波**：支持高斯模糊、中值滤波、均值模糊、双边滤波

### ✨ 图像增强与形态学
- **形态学变换**：支持腐蚀、膨胀、开运算、闭运算
- **直方图均衡化**：增强图像对比度

### 🎯 其他特性
- **实时预览**：参数调整时自动更新处理结果
- **双栏对比**：左侧显示原图，右侧显示处理结果
- **暂存功能**：支持临时保存处理结果，可将暂存图片作为新原图继续处理
- **结果下载**：支持下载处理后的图片
- **多语言支持**：支持中文/英文界面切换

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

```bash
streamlit run streamlit_app.py
```

启动后访问 `http://localhost:8501` 即可使用。

## 📁 项目结构

```
web_ps/
├── streamlit_app.py          # Streamlit 应用主入口
├── requirements.txt          # 依赖列表
├── LICENSE                   # MIT License
├── .streamlit/
│   └── config.toml           # Streamlit 配置文件
├── backend/
│   ├── __init__.py           # 模块初始化
│   ├── image_processor.py    # 图像处理核心逻辑
│   └── models/               # 预训练模型文件
├── deploy/
│   ├── Dockerfile            # Docker 部署配置
│   ├── docker-compose.yml    # Docker Compose 配置
│   ├── nginx.conf            # Nginx 反向代理配置
│   ├── start.sh              # 一键启动脚本
│   └── health_check.py       # 健康检查脚本
├── docs/
│   ├── PRD.md                # 产品需求文档
│   ├── DESIGN.md             # 设计文档
│   ├── API.md                # 接口文档
│   ├── DATABASE.md           # 数据库文档（本项目无数据库）
│   └── TEST_REPORT.md        # 测试报告
├── frontend/                 # 前端代码目录（可扩展）
├── test/
│   └── test_processor.py     # 单元测试
└── screenshots/              # 演示截图
```

## 🔧 使用说明

### 1. 上传图片

点击侧边栏的"上传图片"按钮，选择 JPG 或 PNG 格式的图片。

### 2. 选择模块

在侧边栏选择以下模块之一：
- **检测模块**：边缘检测、人脸检测、目标检测
- **基础图像处理**：色彩空间转换、几何变换、阈值化、平滑/滤波
- **图像增强与形态学**：形态学变换、直方图均衡化

### 3. 选择处理功能

在选中的模块下选择具体功能：

#### 检测模块
- **原图**：显示原始图片
- **边缘检测 (Canny)**：使用 Canny 算法检测边缘
- **人脸检测 (Haar)**：检测图像中的人脸
- **目标检测 (DNN)**：检测图像中的常见物体

#### 基础图像处理模块
- **色彩空间转换**：灰度、HSV、RGB
- **几何变换**：缩放、旋转、平移、仿射变换
- **阈值化**：二值化、反二值化、截断、归零、自适应、Otsu
- **平滑/滤波**：高斯模糊、中值滤波、均值模糊、双边滤波

#### 图像增强与形态学模块
- **形态学变换**：腐蚀、膨胀、开运算、闭运算
- **直方图均衡化**：增强图像对比度

### 4. 调整参数

根据选择的功能，调整对应的参数：

#### 边缘检测
- **阈值1**：低阈值（范围 0-255）
- **阈值2**：高阈值（范围 0-255）

#### 人脸检测
- **缩放因子**：较小的值检测更全面，但可能增加误检
- **最小邻域数**：较大的值减少误检，但可能漏检
- **最小人脸尺寸**：过滤太小的检测结果
- **要求检测到眼睛**：启用后，只有检测到眼睛的区域才会被标记为人脸
- **要求肤色检测**：启用后，只有肤色区域才会被标记为人脸
- **级联分类器置信度**：需要多少个分类器确认才标记为人脸

#### 目标检测
- **置信度阈值**：较低的值检测更多目标，但可能增加误检

### 5. 暂存与下载

- **暂存结果**：点击"暂存"按钮保存当前处理结果
- **使用暂存图片**：点击已保存的图片可将其作为新的原图继续处理
- **下载结果**：点击"下载结果图"按钮，保存处理后的图片

## 🐳 容器化部署

### Docker 单容器部署

```bash
# 构建镜像
docker build -t image-annotation-tool -f deploy/Dockerfile .

# 运行容器
docker run -d -p 8501:8501 image-annotation-tool
```

### Docker Compose 部署（推荐）

```bash
cd deploy
bash start.sh
```

或手动启动：

```bash
cd deploy
docker-compose up -d
```

### 访问应用

启动后访问 `http://localhost:8501` 即可使用。

## ☁️ 云端部署

### 部署到阿里云

#### 架构说明

本项目采用 **ECS + OSS** 架构：
- **ECS（弹性计算服务）**：运行 Streamlit 应用和 Nginx 反向代理
- **OSS（对象存储服务）**：存储上传的图片、处理结果和模型文件（可选）

#### 步骤一：创建 ECS 实例

1. 登录阿里云控制台，进入 ECS 管理页面
2. 创建实例，推荐配置：
   - 操作系统：Ubuntu 22.04 LTS
   - 实例规格：ecs.g6.large（2核8G）或更高
   - 带宽：5M 或更高

#### 步骤二：配置安全组

在 ECS 实例的安全组中添加以下规则：

| 协议 | 端口 | 来源 | 说明 |
|------|------|------|------|
| TCP | 80 | 0.0.0.0/0 | HTTP 访问 |
| TCP | 443 | 0.0.0.0/0 | HTTPS 访问（可选） |
| TCP | 22 | 你的IP | SSH 登录 |

#### 步骤三：安装 Docker 和 Docker Compose

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | bash

# 启动 Docker
systemctl start docker
systemctl enable docker

# 安装 Docker Compose
apt update && apt install docker-compose-plugin -y
```

#### 步骤四：上传项目代码

```bash
# 方式一：使用 Git 克隆（推荐）
git clone <your-repo-url>
cd web_ps

# 方式二：使用 scp 上传压缩包
scp your_project.zip root@your-ecs-ip:/root/
unzip your_project.zip
cd web_ps
```

#### 步骤五：配置 OSS（可选）

**什么是 OSS？**

OSS（Object Storage Service）是阿里云提供的对象存储服务，可用于：
- 持久化存储用户上传的图片
- 存储处理后的结果图片
- 存储模型文件

**创建 OSS Bucket**

1. 登录阿里云控制台，进入 OSS 管理页面
2. 创建 Bucket，推荐配置：
   - Bucket 名称：自定义（如 `image-annotation-bucket`）
   - 地域：选择与 ECS 相同的地域（如 `cn-hangzhou`）
   - 存储类型：标准存储
   - 读写权限：私有（推荐）

**获取 OSS 凭证**

1. 在阿里云控制台进入 RAM 管理页面
2. 创建一个新的 RAM 用户，授予以下权限：
   - `AliyunOSSFullAccess`（或更精细的权限）
3. 获取 AccessKey ID 和 AccessKey Secret

**配置 OSS 环境变量**

编辑 `deploy/docker-compose.yml`，填写 OSS 配置：

```yaml
environment:
  - OSS_ACCESS_KEY_ID=your-access-key-id
  - OSS_ACCESS_KEY_SECRET=your-access-key-secret
  - OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com  # 替换为你的地域
  - OSS_BUCKET_NAME=your-bucket-name
```

#### 步骤六：启动服务

```bash
cd deploy
docker-compose up -d
```

#### 步骤七：验证服务

```bash
# 查看容器状态
docker-compose ps

# 查看实时日志
docker-compose logs -f

# 健康检查
curl http://localhost/health
```

#### 访问应用

在浏览器中访问：`http://你的ECS公网IP`

#### HTTPS 配置（推荐）

**方式一：使用阿里云 SSL 证书**

1. 在阿里云控制台申请免费 SSL 证书
2. 下载证书文件（Nginx 格式）
3. 将证书文件放入 `deploy/ssl/` 目录：

```
deploy/ssl/
├── cert.pem      # 证书文件
└── key.pem       # 私钥文件
```

**方式二：使用 Let's Encrypt**

```bash
# 安装 Certbot
apt install certbot python3-certbot-nginx -y

# 获取证书
certbot certonly --nginx -d your-domain.com

# 更新 Nginx 配置
# 修改 deploy/nginx.conf 添加 HTTPS 配置
```

#### 部署注意事项

1. **镜像加速**：阿里云 ECS 默认配置了 Docker 镜像加速器，无需额外配置
2. **pip 镜像**：项目已配置使用清华镜像源，适合国内环境
3. **模型文件**：项目已将模型文件打包进 Docker 镜像，无需额外下载
4. **日志持久化**：日志会存储在 `logs/` 目录，建议定期清理或配置日志轮转

### 部署到腾讯云

部署步骤与阿里云类似：

1. 创建 CVM 实例（Ubuntu 22.04 LTS）
2. 配置安全组开放 80、443、22 端口
3. 安装 Docker 和 Docker Compose
4. 上传项目代码
5. 启动服务

```bash
cd deploy
docker-compose up -d
```

## 🧪 运行测试

```bash
cd test
pytest test_processor.py -v
```

## 📝 API 文档

详细的接口文档请查看 [docs/API.md](docs/API.md)。

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。
