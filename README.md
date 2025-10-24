# EMMET Stability Replay

Stability-focused extensions to EMMET model editing workflows.

## 项目结构

```
llm_project/  # 项目名称
├── LICENSE  # 开源许可证（如 MIT 或 Apache-2.0）
├── README.md  # 项目概述、安装指南和使用示例
├── pyproject.toml  # 项目配置：管理依赖、构建、工具（如 black、pytest）
├── Makefile  # 自动化任务脚本（make env, make install, make test）
├── .gitignore  # 排除 data/、models/、.venv/ 等大型文件
├── data/  # 数据目录
│   ├── raw/  # 原始数据集（如文本语料）
│   ├── processed/  # 处理后的数据（如 tokenized 数据集）
│   └── embeddings/  # 预计算的向量嵌入
├── configs/  # 配置目录
│   ├── model.yaml  # LLM 参数（如温度、top_p）
│   └── prompts/  # 提示模板文件
├── models/  # 模型目录
│   ├── checkpoints/  # 微调后的模型权重
│   └── download_scripts/  # 下载预训练模型的脚本
├── src/  # 源代码包
│   ├── __init__.py  # 包初始化
│   ├── main.py  # 入口脚本
│   ├── core.py  # 核心逻辑
│   ├── data/  # 数据处理模块
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/  # 特征工程模块
# EMMET Stability Replay

Stability-focused extensions to EMMET model editing workflows.

## 项目结构

### 当前结构

```
emmet-stability-replay/
├── Makefile              # 自动化构建脚本
├── pyproject.toml        # 项目配置和依赖
├── README.md
├── src/                  # 源代码（扁平结构）
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── test/                 # 测试目录
└── scripts/              # 辅助脚本
```

### 推荐的完整 LLM 项目结构（参考）

```
llm_project/
├── LICENSE
├── README.md
├── pyproject.toml        # 项目配置：依赖、构建工具
├── Makefile              # 自动化任务：make env, make install, make test
├── .gitignore
├── data/                 # 数据目录（不提交到 Git）
│   ├── raw/             # 原始数据集
│   ├── processed/       # 处理后的数据
│   └── embeddings/      # 预计算的向量
├── configs/              # 配置文件
│   ├── model.yaml       # 模型参数（温度、top_p）
│   └── prompts/         # 提示模板
├── models/               # 模型文件
│   ├── checkpoints/     # 微调后的权重
│   └── download_scripts/
├── src/                  # 源代码包
│   ├── __init__.py
│   ├── main.py
│   ├── core.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_embeddings.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── train_model.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── huggingface.py
│   │   └── openai.py
│   └── utils/
│       ├── __init__.py
│       └── logging.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── conftest.py
├── docs/
│   └── api.md
└── notebooks/
    └── exploration.ipynb
```

## 环境要求

**所有平台：**
- 安装 [Miniforge](https://github.com/conda-forge/miniforge)（推荐，自带 mamba）

**Linux / macOS：**
- ✅ 系统自带 make

**Windows：**
- 安装 GNU Make：
  ```powershell
  choco install make
  # 或
  winget install GnuWin32.Make
  ```

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url>
cd emmet-stability-replay

# 2. 创建环境（自动检测 GPU/CPU）
make env

# 3. 安装项目（开发模式）
make install

# 4. 激活环境
conda activate emmet-edit

# 5. 验证安装
python -c "from data import make_dataset; print('✅ 安装成功')"
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `make env` | 创建/更新环境（自动检测 GPU/CPU） |
| `make env-cpu` | 强制 CPU 版本 |
| `make env-gpu` | 强制 GPU 版本 |
| `make install` | 开发模式安装（代码修改立即生效） |
| `make test` | 运行测试 |
| `make build` | 构建发行包 |
| `make clean` | 清理构建产物 |

## 开发说明

- **开发模式**：`make install` 后修改代码无需重新安装
- **模块导入**：`from data import make_dataset`（简洁的扁平结构）
- **环境变量**：`DEVICE=cpu make env` 手动指定设备
- **打包发布**：`make build` 生成 wheel 包到 `dist/` 目录