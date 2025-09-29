# README

这是一个基于 Hugo + LoveIt 主题 的博客仓库模版（多语言），用于部署到 GitHub Pages。

主要目标：
- 首页展示英文内容（默认语言为 en）
- 每篇文章可以拥有对应的中文翻译（文件名后缀 .en.md / .zh.md），Hugo 会自动在文章页显示语言切换和翻译链接

快速上手：

1. 安装 Hugo（建议使用 Hugo Extended）。
2. 在仓库根目录添加 LoveIt 主题：

	git submodule add https://github.com/dillonzq/LoveIt themes/LoveIt

	或者手动下载并放到 `themes/LoveIt` 下。

3. 本地运行开发服务器：

```bash
hugo server -D
```

4. 构建并部署到 GitHub Pages：仓库示例包含一个 GitHub Actions 工作流，会在推到 `main` 分支时将生成的 `public/` 内容发布到 `gh-pages` 分支。你需要在仓库 `Settings > Secrets` 中设置 `ACTIONS_DEPLOY_KEY` 或使用默认 workflow 的配置说明。

重要说明：此仓库模板未包含 LoveIt 主题代码（体积较大），请按上面第 2 步将主题添加到 `themes/LoveIt`，否则构建会失败。

示例文件：
- `config.toml`：多语言配置（en 默认），主题设置
- `content/post/hello-world.en.md`：英文示例文章
- `content/post/hello-world.zh.md`：对应的中文翻译
- `.github/workflows/gh-pages.yml`：自动构建并部署到 gh-pages 分支（需在仓库中启用 Actions）

如果需要，我可以：
- 帮你把 LoveIt 主题作为子模块添加（需要允许网络访问或提供主题代码）
- 根据 LoveIt 的特性定制首页、社交链接和菜单
