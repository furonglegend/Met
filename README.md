# EMMET Stability Replay

Stability-focused extensions to EMMET model editing workflows.

## 项目结构

```plaintext
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
│   ├── main.py
│   ├── core.py
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_embeddings.py
│   ├── models/
│   │   └── train_model.py
│   ├── integrations/
│   │   ├── huggingface.py
│   │   └── openai.py
│   └── utils/
│       └── logging.py
├── tests/
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

## 依赖管理

### 使用uv

```bash
# 新增包
uv add <package-name>

# 新增特定版本包
uv add <package-name>==<version>

# 移除包
uv remove <package-name>
```

### 不使用uv

添加时使用 pip 或 conda 手动管理依赖，并手动添加到 `pyproject.toml` 的 `dependencies` 部分，删除同理
