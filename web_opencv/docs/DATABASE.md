# Web端图像标注工具 - 数据库文档

## 概述

本项目为纯图像处理工具，不涉及数据库存储功能。所有图像处理均在内存中完成，处理结果通过文件下载方式保存到本地。

## 无数据库原因

1. **数据特性**：图像处理结果为临时数据，用户通过下载方式获取，无需持久化存储
2. **架构设计**：采用前后端一体的 Streamlit 架构，无需数据库支撑
3. **简化部署**：无数据库依赖，便于容器化部署和云平台集成

## 如需扩展数据库功能（可选）

如果未来需要添加图片历史记录、用户管理等功能，建议使用以下方案：

### 数据库选择

| 数据库类型 | 推荐方案 | 适用场景 |
| :--- | :--- | :--- |
| 关系型数据库 | SQLite / PostgreSQL | 用户管理、处理历史记录 |
| 对象存储 | AWS S3 / 阿里云 OSS | 图片存储 |

### 表结构设计（示例）

**users 表**：
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| id | INTEGER | 主键 |
| username | VARCHAR(50) | 用户名 |
| email | VARCHAR(100) | 邮箱 |
| created_at | TIMESTAMP | 创建时间 |

**processing_history 表**：
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| id | INTEGER | 主键 |
| user_id | INTEGER | 用户ID（外键） |
| original_image_path | VARCHAR(255) | 原图路径 |
| processed_image_path | VARCHAR(255) | 处理结果路径 |
| function_type | VARCHAR(20) | 处理功能类型 |
| parameters | JSON | 参数配置 |
| created_at | TIMESTAMP | 创建时间 |

---

*本项目当前版本无数据库依赖，此文档仅供未来扩展参考。*
