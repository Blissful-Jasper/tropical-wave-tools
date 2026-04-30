# 部署发布

## 发布到 GitHub

```bash
git init
git add .
git commit -m "Initial modernized package scaffold"
git branch -M main
git remote add origin git@github.com:YOUR_NAME/tropical-wave-tools.git
git push -u origin main
```

## GitHub Pages 文档

推荐使用 GitHub Actions 自动发布，而不是本地直接推送 `gh-pages`。

1. 在 GitHub 仓库设置里把 Pages source 切换为 `GitHub Actions`。
2. 推送 `main` 或当前开发分支。
3. 等待 `Pages` workflow 完成后自动更新站点。

当前仓库已经提供自动部署工作流：

```bash
git push origin apple
```

如果你只想本地重建网页，不部署：

```bash
bash scripts/twave.sh docs
```

如果你仍然想尝试本地直接推送 `gh-pages`：

```bash
bash scripts/twave.sh pages
```

但这种方式依赖本地 Git HTTP/SSH 推送稳定性，可靠性通常不如 GitHub Actions。

## Streamlit 演示

```bash
bash scripts/twave.sh app
```

## PyPI

```bash
bash scripts/twave.sh build
bash scripts/twave.sh publish
```
