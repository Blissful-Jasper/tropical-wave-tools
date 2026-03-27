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
pip install -e ".[docs]"
python scripts/generate_gallery.py
mkdocs gh-deploy
```

## Streamlit 演示

本项目同时包含一个轻量交互页：

```bash
pip install -e ".[app]"
streamlit run apps/streamlit_app.py
```

如果需要在线部署，可以推送到 Streamlit Community Cloud，入口文件为：

```text
apps/streamlit_app.py
```

