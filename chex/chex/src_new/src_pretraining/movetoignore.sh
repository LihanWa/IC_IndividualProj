#!/bin/bash

# 定义大小阈值：此处设置为 5MB
size_threshold=5

# 创建.gitignore文件（如果不存在）
touch .gitignore

# 使用find命令查找大于阈值的文件并正确处理
find . -type f -size +${size_threshold}M | while read -r file; do
    # 去除前缀 ./，方便记录在 .gitignore 中
    path=${file#./}
    echo "将文件 '$path' 添加到 .gitignore 中..."
    
    # 先检查是否已经存在 .gitignore 中
    if ! grep -qxF "$path" .gitignore; then
        echo "$path" >> .gitignore
    fi

    # 如果文件已被 Git 跟踪，则将其从暂存区移除（保留工作区）
    git ls-files --error-unmatch "$path" &>/dev/null && git rm --cached "$path"
done
