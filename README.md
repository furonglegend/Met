## 项目结构

```
project_name/
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── external/         # 外部数据源
├── notebooks/            # Jupyter notebooks，用于探索
│   └── exploration.ipynb
├── src/                  # 源代码目录
│   ├── __init__.py
│   ├── data/             # 数据处理模块
│   │   └── make_dataset.py
│   ├── features/         # 特征工程
│   │   └── build_features.py
│   ├── models/           # 模型训练
│   │   └── train_model.py
│   └── visualization/    # 可视化模块
│       └── visualize.py
├── models/               # 训练模型输出
├── config/               # 配置目录（YAML或TOML文件）
│   └── config.yaml
├── reports/              # 报告和图表
│   └── figures/
├── tests/                # 测试目录
├── pyproject.toml        # 项目配置
├── README.md             # 项目说明
├── LICENSE               # 许可证
└── .gitignore            # Git忽略文件

```

## 脚本使用

- 环境配置：在项目根目录执行 `bash scripts/setup_env.sh`，脚本会自动检测是否存在 `mamba`，优先使用其创建/更新 Conda 环境；若系统不支持 `mamba` 则回退至 `conda`。可通过环境变量 `DEVICE=cpu` 或 `DEVICE=gpu` 手动覆盖默认的设备检测。
- 项目打包：执行 `bash scripts/build_package.sh`，脚本会优先使用 `uv build` 生成发行包；若系统未安装 `uv`，则自动回退至 `python -m build`。