# 推送到 GitHub 的步骤

仓库内容已经整理好，并且**本地仓库已初始化、68 个文件已暂存**（0.90 MB），
你只需要 commit + push 两步。

## 一、先在 GitHub 建空仓库

1. 登录 GitHub → New repository
2. Repository name 填：`UA-DETRAC-YOLOv8n-ASP`
3. **不要**勾选 "Add a README file" / ".gitignore" / "license"（本地已有，避免冲突）
4. 创建后地址为：`https://github.com/kzzzt/UA-DETRAC-YOLOv8n-ASP.git`

## 二、本地提交并推送

在本机打开 PowerShell，执行：

```powershell
cd "C:\Users\康智童\OneDrive\桌面\yolo\UA-DETRAC-YOLOv8n-ASP"

# 建议先把提交作者名改成与 GitHub 一致的 kzzzt
# （当前全局配置是 "kzzzt~"，末尾多了一个波浪号）
git config user.name "kzzzt"

git commit -m "Code and evaluation artefacts for the YOLOv8n-ASP study on UA-DETRAC"
git branch -M main
git remote add origin https://github.com/kzzzt/UA-DETRAC-YOLOv8n-ASP.git
git push -u origin main
```

> 首次推送会弹出 GitHub 登录窗口（推荐浏览器授权或 Personal Access Token）。

## 三、已完成的自检（无需你做）

| 检查项 | 结果 |
|--------|------|
| `.gitignore` 是否漏掉该排除的 | ✅ `*.pt`、`*.jpg`、`runs/`、`__pycache__`、临时脚本全部排除 |
| 是否误伤了必需文件 | ✅ 已修两个缺陷：`_*.py` 曾误伤 `models/__init__.py` 等 3 个包初始化文件；`!docs/figures/*.png` 曾漏掉 `paper/figures/*.png`。现 68 个文件全部正确 |
| 仓库体积 | ✅ 0.90 MB（68 个文件）|
| 论文中的仓库地址 | ✅ `paper/main.tex` 的 Code availability 与 `paper/cover_letter.md` 均为 `https://github.com/kzzzt/UA-DETRAC-YOLOv8n-ASP`，推送后无需再改 |

## 四、推送后自检

```powershell
git ls-files | Select-String "results/|scripts/|paper/main.tex|README.md|LICENSE"
```

浏览器打开仓库页面，确认：

- [ ] README 正常渲染（标题、结果表、命令块）
- [ ] `results/` 里 15 个 CSV/MD 都在
- [ ] `docs/figures/` 与 `paper/figures/` 各三张图能预览
- [ ] `paper/main.tex` 可在线查看
- [ ] 仓库体积小于 5 MB

## 五、可选：加一个 Release

投稿时可以在 GitHub 上打一个 tag 作为"论文对应版本"，审稿人引用更明确：

```powershell
git tag -a v1.0-applied-intelligence -m "Version corresponding to the Applied Intelligence submission"
git push origin v1.0-applied-intelligence
```
