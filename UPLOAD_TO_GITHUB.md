# 上传到 GitHub 的步骤说明

## 注意事项

**重要**：`LaViT-15k/` 目录（~150GB）已通过 `.gitignore` 排除，**不会**上传到 GitHub。

GitHub 仓库只包含：
- ✅ 代码（data_construction, training, evaluation 等）
- ✅ 文档（README.md, LICENSE 等）
- ✅ 配置文件（requirements.txt, .gitignore 等）
- ❌ LaViT-15k 数据集（太大，需要外部存储）

## 上传步骤

### 1. 初始化 Git 仓库（如果还没有）

```bash
cd /root/autodl-tmp/LaViT_OpenSource
git init
```

### 2. 添加远程仓库

```bash
git remote add origin https://github.com/1324fgg/Quan.git
# 或者如果已经存在，更新远程地址：
# git remote set-url origin https://github.com/1324fgg/Quan.git
```

### 3. 确认 .gitignore 配置

确认 `.gitignore` 文件中包含：
```
LaViT-15k/
```

这确保整个 LaViT-15k 目录不会被提交。

### 4. 添加文件并提交

```bash
# 添加所有文件（LaViT-15k 会被自动忽略）
git add .

# 查看将要提交的文件（确认 LaViT-15k 不在其中）
git status

# 提交
git commit -m "Initial commit: LaViT OpenSource implementation"
```

### 5. 推送到 GitHub

```bash
# 如果远程仓库是空的，直接推送
git push -u origin main

# 或者如果远程仓库使用 master 分支
git push -u origin master
```

如果遇到错误（比如远程仓库已有内容），可能需要先拉取：

```bash
# 先拉取远程内容（如果存在）
git pull origin main --allow-unrelated-histories

# 然后再推送
git push -u origin main
```

## 验证上传

上传完成后，访问 https://github.com/1324fgg/Quan 确认：

1. ✅ README.md 已上传
2. ✅ 代码目录（data_construction, training, evaluation 等）已上传
3. ✅ LaViT-15k 目录**不在**仓库中

## LaViT-15k 数据集处理

LaViT-15k 数据集（~150GB）应该：

1. **上传到外部存储**（推荐 Hugging Face Datasets 或 Zenodo）
2. **在 README 中添加下载链接**
3. **不在 GitHub 仓库中**

可以参考 `LaViT-15k/DATASET_STORAGE.md` 了解数据集存储方案。
