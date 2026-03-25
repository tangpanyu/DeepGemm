# CUDA 开发环境配置指南

## 📋 问题说明

在 IDE (如 CLion, VSCode) 中开发 CUDA 项目时，可能会遇到无法索引到 CUDA 头文件的问题，例如：
- `cuda.h` 找不到
- `cutlass/cutlass.h` 无法识别
- CUDA 内置函数和类型报错

## ✅ 已完成的配置

### 1. `.clangd` 配置文件

已在项目根目录创建 `.clangd` 文件，包含：
- ✅ CUDA 编译标志 (`-DCUDA_ARCH=900`, `-D__CUDACC__`)
- ✅ CUDA 头文件路径 (`/usr/local/cuda/include`)
- ✅ CUDA 设备代码支持 (`-xcuda`, `--cuda-gpu-arch=sm_90`)

### 2. `compile_commands.json` 符号链接

已创建符号链接指向 build 目录中的编译数据库：
```bash
compile_commands.json -> build/compile_commands.json
```

该文件已包含完整的 CUDA 路径：
- `-I/usr/local/cuda/include`
- `-isystem /usr/local/cuda/targets/x86_64-linux/include`

## 🔧 使用方式

### CLion IDE

1. **打开设置**: `File → Settings → Build, Execution, Deployment → CMake`
2. **设置编译数据库路径**: 
   - 编译数据库路径：`/home/tangpanyu/reps/DeepGEMM/build`
3. **重新加载 CMake 项目**
4. **重启 clangd 服务**:
   - `File → Invalidate Caches / Restart`
   - 或者手动删除 `.idea` 目录后重新打开项目

### VSCode + clangd 插件

1. **安装 clangd 插件**
2. **配置 `.vscode/settings.json`**:
```json
{
    "clangd.path": "/usr/bin/clangd",
    "clangd.arguments": [
        "--compile-commands-dir=build",
        "--query-driver=/usr/bin/g++,/usr/bin/clang++",
        "--background-index"
    ]
}
```

3. **重新加载窗口**: `Ctrl+Shift+P → Developer: Reload Window`

### Neovim/Vim + coc.nvim

1. **确保 `.clangd` 文件在项目根目录**
2. **coc.nvim 会自动读取 `compile_commands.json`**
3. **重启 coc 服务**: `:CocRestart`

## 🚀 验证配置

### 方法 1: 检查 clangd 日志

在 IDE 中打开任意 CUDA 文件 (`.cu` 或 `.cuh`)，查看输出面板：
```
I[timestamp]: Loaded compilation database from /path/to/compile_commands.json
I[timestamp]: ASTWorker building file /path/to/file.cu
```

### 方法 2: 测试 CUDA 头文件索引

打开任意 kernel 文件，尝试：
- 跳转到 `cudaMalloc` 定义
- 补全 `cudaDeviceProp` 类型
- 查看 `__global__` 宏定义

如果都能正常工作，说明配置成功！

### 方法 3: 手动测试 clangd

```bash
# 测试 clangd 是否能找到 CUDA 头文件
clangd --check=/home/tangpanyu/reps/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh
```

输出应该包含：
```
Parsing command...
-I/usr/local/cuda/include
-isystem /usr/local/cuda/targets/x86_64-linux/include
...
```

## 🔍 常见问题

### Q1: 仍然提示找不到 `cuda.h`

**解决方案**:
1. 确认 CUDA 安装路径：
   ```bash
   ls -la /usr/local/cuda/include/cuda.h
   ```
2. 如果使用 conda 环境，可能需要：
   ```bash
   export CPATH=/usr/local/cuda/include:$CPATH
   ```
3. 在 `.clangd` 中添加绝对路径：
   ```yaml
   compileFlags:
     - -I/usr/local/cuda/include
   ```

### Q2: CUTLASS 头文件找不到

**解决方案**:
1. 确认 third-party 子模块已初始化：
   ```bash
   git submodule update --init --recursive
   ```
2. 检查 `.clangd` 中是否包含 CUTLASS 路径：
   ```yaml
   compileFlags:
     - -I/home/tangpanyu/reps/DeepGEMM/third-party/cutlass/include
   ```

### Q3: 代码跳转不工作

**解决方案**:
1. 清除 clangd 缓存：
   ```bash
   rm -rf .cache/clangd
   ```
2. 重新构建索引：
   - CLion: `File → Invalidate Caches / Restart`
   - VScode: `Ctrl+Shift+P → clangd: Restart clangd`

### Q4: 智能补全很慢

**解决方案**:
1. 启用背景索引：
   ```yaml
   Index:
     Background: Build
   ```
2. 减少索引范围（可选）：
   ```yaml
   Diagnostics:
     SuppressedPaths:
       - "**/third-party/**"
   ```

## 📝 更新配置后的操作

每次修改 `.clangd` 或 `compile_commands.json` 后：

1. **CLion**: 
   - `File → Reload CMake Project`
   - 或 `File → Invalidate Caches / Restart`

2. **VScode**:
   - `Ctrl+Shift+P → clangd: Restart clangd`

3. **Vim/Neovim**:
   - `:CocRestart` 或 `:LspRestart`

## 🎯 推荐的 IDE 设置

### CLion

```
Settings → Build, Execution, Deployment → Toolchains
- CMake: 使用系统 CMake 3.26+
- C Compiler: gcc
- C++ Compiler: g++

Settings → Languages & Frameworks → C/C++
- Code Style: LLVM
- Formatter: ClangFormat
```

### VScode

```json
{
    "C_Cpp.default.configurationProvider": "llvm-vs-code-extensions.vscode-clangd",
    "clangd.arguments": [
        "--background-index",
        "--clang-tidy",
        "--header-insertion=iwyu",
        "--completion-style=detailed"
    ],
    "files.associations": {
        "*.cuh": "cuda-cpp",
        "*.cu": "cuda-cpp"
    }
}
```

## 📚 参考资料

- [clangd 官方文档](https://clangd.llvm.org/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS Documentation](https://nvidia.github.io/cutlass/)

---

**最后更新**: 2026-03-25
**适用项目**: DeepGEMM (SM90/SM100 CUDA Kernels)
