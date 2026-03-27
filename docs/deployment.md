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

```bash
bash scripts/twave.sh pages
```

## Streamlit 演示

```bash
bash scripts/twave.sh app
```

## PyPI

```bash
bash scripts/twave.sh build
bash scripts/twave.sh publish
```
