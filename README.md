# Kagura Tart Blog

个人博客，基于 Astro 5 静态生成。默认语言为中文，保留英文翻译。

## 目录结构

```text
src/
├── components/              # 页面组件
├── content/blog/
│   ├── YYYY/MM/*.md         # 中文文章（默认语言）
│   └── en/YYYY/MM/*.md      # 英文翻译（由脚本生成）
├── i18n/                    # 语言配置与 slug 工具
├── layouts/
├── pages/
└── config.ts                # 首页个人信息
scripts/                     # 翻译、i18n 覆盖率与构建检查
public/                      # 静态资源（图片等）
```

## 文章规则

- 中文文章放在 `src/content/blog/<年>/<月>/`，英文翻译放在 `src/content/blog/en/<年>/<月>/`。
- **日期目录只用于本地归档，不进入 URL。** 文章 URL 只取文件名：
  - `src/content/blog/2026/04/foo.md` → `/blog/foo`
  - `src/content/blog/en/2026/04/foo.md` → `/en/blog/foo`
- 因此**所有文章的文件名必须全局唯一**；重名会在构建时直接报错。
- 图片请使用 `/blog-assets/...` 或 `/...` 绝对路径，不要用相对路径，避免移动文件后失效。

## Frontmatter

```yaml
---
title: "文章标题"
description: "摘要"
pubDate: 2026-04-27
updatedDate: 2026-05-01   # 可选
tags: ["UAV", "VLM"]
category: "Tech"          # Tech | Essay | Paper
draft: false              # 可选
---
```

## 新增或更新文章

1. 在中文目录新建或修改 md。
2. 运行 `npm run translate:i18n` 生成或更新英文翻译（使用 Google 翻译接口并缓存，代码块会跳过）。
3. 检查 `src/content/blog/en/...` 下的翻译，必要时手动修正。
4. 运行 `npm run build` 和 `npm run check:i18n` 验证。
5. 提交时确保中文源文和英文翻译一起 stage。

> 翻译文件带有 `sourceHash`；源文修改后重新运行脚本会更新对应翻译。不要手动删除这个字段。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `npm run dev` | 启动本地开发服务器 |
| `npm run build` | 构建静态站点到 `dist/` |
| `npm run translate:i18n` | 生成或更新英文翻译 |
| `npm run check:i18n:coverage` | 检查每篇中文文章是否有最新英文翻译 |
| `npm run check:i18n` | 检查构建后的中英文路由是否完整 |
| `npm run test:i18n:coverage` | i18n 覆盖率脚本单元测试 |
| `npm run preview` | 预览构建结果 |

## 语言策略

当前只保留中文（默认）和英文。`ja`、`fr`、`de` 的机器翻译已移除，以降低内容冗余和维护成本。如需恢复其他语言，在 `src/i18n/config.mjs` 的 `locales` 中添加语言代码，然后重新运行翻译脚本即可。
