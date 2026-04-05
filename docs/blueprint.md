# 改版蓝图

## 站点角色

| 页面 | 主要任务 |
|---|---|
| 首页 | 结果预览与核心入口 |
| Quick Start | 最短安装与最短代码 |
| Gallery | 旗舰案例与结果图 |
| API | 模块与核心函数 |
| Methods | 方法流程与原理 |
| Notes | 研究记录与实现笔记 |

## 视觉

- 主色：`#0f766e`
- 强调色：`#0284c7`
- 文字色：`#0f172a`
- 边框：`#cbd5e1`
- 背景：白色或浅灰蓝白

## 旗舰案例

- 样例 OLR 预览
- 异常场标准差
- WK 频谱
- Kelvin 滤波 Hovmöller
- Legacy vs CCKW 对比

## 目录方向

- `docs/`：稳定展示页
- `examples/`：最短可运行示例
- `notebooks/`：长教程与案例
- `notes/`：研究记录
- `src/tropical_wave_tools/`：核心包

## 迁移方向

1. 当前阶段保留 `mkdocs`
2. 后续迁移到 `Sphinx + PyData Sphinx Theme + sphinx-gallery`
