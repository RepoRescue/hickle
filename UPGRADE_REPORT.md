# 升级报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 仓库名 | hickle |
| 升级时间 | 2026-03-13 |
| 升级状态 | ✅ 成功 |

## Python 版本

| 升级前 | 升级后 |
|--------|--------|
| >=3.7 | >=3.13 |

## 依赖变更

### 核心依赖

| 依赖 | 升级前 | 升级后 |
|------|--------|--------|
| h5py | >=2.10.0 | ==3.16.0 |
| numpy | >=1.8,!=1.20 | ==2.4.3 |

### 测试依赖

| 依赖 | 升级前 | 升级后 |
|------|--------|--------|
| dill | >=0.3.0 | ==0.4.1 |
| pytest | >=4.6.0 | ==9.0.2 |
| pytest-cov | (无版本) | ==7.0.0 |
| pytest-timeout | (新增) | ==2.4.0 |
| astropy | (无版本) | ==7.2.0 |
| scipy | >=1.0.0 | ==1.17.1 |
| pandas | >=0.24.0 | ==3.0.1 |
| codecov | (无版本) | ==2.1.13 |
| check-manifest | (无版本) | ==0.51 |
| twine | >=1.13.0 | ==6.2.0 |

## 代码修改

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| hickle/loaders/load_builtins.py:388 | NumPy 2.0 适配 | `np.array(copy=False)` → `np.asarray()` |
| hickle/loaders/load_numpy.py:235 | NumPy 2.0 适配 | `np.array(copy=False)` → `np.asarray()` |
| hickle/tests/test_06_load_astropy.py:169,177 | NumPy 2.0 适配 | `.tostring()` → `.tobytes()` |
| hickle/legacy_v3/hickle.py:28 | Python 3.13 适配 | `pkg_resources` → `importlib.metadata` |
| setup.py:55-59 | Python 版本声明 | 移除 3.7-3.11，添加 3.13 |
| setup.py:66 | Python 版本要求 | `>=3.7` → `>=3.13` |

## 测试结果

| 测试类型 | 结果 |
|----------|------|
| 升级前 | N/A (未在旧环境测试) |
| 升级后 | ✅ 102 passed, 0 failed |

## 主要兼容性问题及解决方案

### 1. NumPy 2.0 `copy` 参数变更

**问题**: NumPy 2.0 移除了 `np.array(copy=False)` 的支持

**解决方案**: 使用 `np.asarray()` 替代，它会在需要时自动复制

**影响文件**:
- `hickle/loaders/load_builtins.py`
- `hickle/loaders/load_numpy.py`

### 2. NumPy `tostring()` 方法移除

**问题**: NumPy 2.0 移除了 `ndarray.tostring()` 方法

**解决方案**: 使用 `tobytes()` 替代（功能相同）

**影响文件**:
- `hickle/tests/test_06_load_astropy.py`

### 3. `pkg_resources` 模块移除

**问题**: Python 3.12+ 移除了 `pkg_resources` 模块

**解决方案**: 使用 `importlib.metadata` 替代

**影响文件**:
- `hickle/legacy_v3/hickle.py`

## 备注

- 所有测试均通过，无需修改测试逻辑
- 代码覆盖率达到 100%
- 升级过程中未发现需要人工干预的问题
- 依赖版本已锁定，确保可重现构建
