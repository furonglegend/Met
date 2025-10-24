# __init__.py 维护指南

## 何时需要 __init__.py？

每个需要作为 Python 包导入的目录都需要一个 `__init__.py` 文件。

## 维护规则

### 1. 添加新的 Python 模块文件时

假设您在 `src/data/` 下添加了新文件 `loader.py`：

```python
# src/data/loader.py
class DataLoader:
    def load(self, path):
        pass
```

**更新 `src/data/__init__.py`**：

```python
# src/data/__init__.py
"""Data processing module"""

from .make_dataset import make_dataset, preprocess_text
from .loader import DataLoader  # ← 添加新的导入

__all__ = ["make_dataset", "preprocess_text", "DataLoader"]  # ← 添加到导出列表
```

### 2. 添加新的包目录时

假设您创建了新目录 `src/models/`：

```
src/models/
├── __init__.py      # ← 需要创建这个文件
└── trainer.py
```

**创建 `src/models/__init__.py`**：

```python
# src/models/__init__.py
"""Model training and inference module"""

from .trainer import Trainer

__all__ = ["Trainer"]
```

### 3. __init__.py 的内容模板

#### 最简版本（空包）
```python
"""Package description"""
```

#### 标准版本（推荐）
```python
"""Package description"""

from .module1 import func1, func2
from .module2 import Class1

__all__ = ["func1", "func2", "Class1"]
```

#### 完整版本（根包）
```python
"""Package description"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .module1 import func1
from .module2 import Class1

__all__ = ["func1", "Class1"]
```

## 实际例子

### 当前项目结构
```
src/
├── __init__.py              # 项目根包元数据
├── data/
│   ├── __init__.py          # 导入 make_dataset.py 的函数
│   └── make_dataset.py
├── features/
│   ├── __init__.py          # 暂时为空，等待添加功能
│   └── (待添加功能文件)
└── utils/
    ├── __init__.py          # 暂时为空，等待添加功能
    └── (待添加工具文件)
```

### 工作流程

1. **创建新模块文件**：`src/utils/logging.py`
   ```python
   def setup_logger(name):
       pass
   ```

2. **更新对应的 __init__.py**：`src/utils/__init__.py`
   ```python
   from .logging import setup_logger
   __all__ = ["setup_logger"]
   ```

3. **现在可以简洁导入**：
   ```python
   from utils import setup_logger  # ✅
   ```

## 常见问题

**Q: 是否每个目录都需要 __init__.py？**
A: Python 3.3+ 不强制要求，但建议添加以便简洁导入。

**Q: __all__ 必须吗？**
A: 非必需，但推荐使用。它明确定义了 `from package import *` 导入什么。

**Q: 忘记更新 __init__.py 会怎样？**
A: 需要使用完整路径导入：`from data.make_dataset import make_dataset`

**Q: __init__.py 可以包含业务代码吗？**
A: 可以，但不推荐。应该只包含导入语句和包元数据。
