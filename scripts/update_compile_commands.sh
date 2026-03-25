#!/bin/bash
# 更新 compile_commands.json 以包含完整的 CUDA 路径

BUILD_DIR="/home/tangpanyu/reps/DeepGEMM/build"
COMPILE_CMDS="$BUILD_DIR/compile_commands.json"

# 检查是否安装了 jq (JSON 处理工具)
if ! command -v jq &> /dev/null; then
    echo "错误：需要安装 jq 工具来处理 JSON"
    echo "请运行：sudo apt-get install jq"
    exit 1
fi

# 备份原文件
cp "$COMPILE_CMDS" "${COMPILE_CMDS}.bak"

# 更新 compile_commands.json，添加额外的编译标志
jq '
.[] |= .command += " -I/usr/local/cuda/include -isystem /usr/local/cuda/targets/x86_64-linux/include"
' "$COMPILE_CMDS" > "${COMPILE_CMDS}.tmp" && mv "${COMPILE_CMDS}.tmp" "$COMPILE_CMDS"

echo "✓ compile_commands.json 已更新"
echo "✓ 添加了 CUDA 包含路径:"
echo "  - /usr/local/cuda/include"
echo "  - /usr/local/cuda/targets/x86_64-linux/include"
echo ""
echo "提示：请在 IDE 中重新加载项目或重启 clangd 服务"
