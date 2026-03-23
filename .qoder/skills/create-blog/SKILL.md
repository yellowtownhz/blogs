---
name: create-blog
description: Convert a Chinese research article from the Writing folder into a bilingual (Chinese/English) Hugo blog post in the posts directory. Handles translation, wiki-style link conversion to academic citations, and proper frontmatter formatting.
---

# Create Blog Post

## Purpose

Convert a Chinese research article from the Writing folder into a bilingual Hugo blog post with proper formatting, translation, and academic-style citations.

## Workflow

### Step 1: Read Source Article

Read the source `.md` file from `/root/workspace/note/01_Research/Writing/`.

### Step 2: Create Directory Structure

Create a new directory under `/root/workspace/code/blogs/content/posts/`:
- Directory name: Use kebab-case version of the article title (e.g., `self-distillation-trilogy`)
- Create `index.zh.md` and `index.en.md` files

### Step 3: Process Content

#### Frontmatter (YAML)

Chinese version (`index.zh.md`):
```yaml
---
title: "中文标题"
date: YYYY-MM-DDTHH:MM:SS+08:00
tags: ["标签1", "标签2"]
categories: ["分类"]
draft: false
summary: "中文摘要"
author: "黄镇"
lang: "zh"
type: "posts"
resources:
- name: "featured-image"
  src: "cover.png"  # if images exist
---
```

English version (`index.en.md`):
```yaml
---
title: "English Title"
date: YYYY-MM-DDTHH:MM:SS+08:00
tags: ["Tag1", "Tag2"]
categories: ["Category"]
draft: false
summary: "English summary"
author: "yellowtown"
lang: "en"
type: "posts"
resources:
- name: "featured-image"
  src: "cover.png"  # if images exist
---
```

#### Content Processing Rules

1. **Remove wiki-style links**: Convert `[[Link_Name]]` to plain text `Link_Name`
2. **Convert to academic citations**: Replace wiki links with numbered citations like `[1]`, `[2]`, etc.
3. **Translate content**: Translate Chinese content to English while maintaining technical accuracy
4. **Preserve structure**: Keep headers, lists, tables, code blocks, and admonitions intact
5. **Handle images**: If the source has images, copy them to the post directory and reference correctly

#### Admonition Format

Use Hugo admonition shortcode for notes/reflections:

```markdown
{{< admonition note "个人思考" >}}
Content here
{{< /admonition >}}
```

English version:
```markdown
{{< admonition note "Reflection" >}}
Content here
{{< /admonition >}}
```

#### Code Block Best Practices

**IMPORTANT**: Avoid using plain text code blocks (``` without language specifier) for content that should be rendered as formatted text. Hugo's Markdown renderer may not handle them correctly.

**Instead, use admonition boxes for examples/scenarios:**

```markdown
{{< admonition example "示例" >}}
**问题**: "问题内容"  
**演示**: "演示内容"

**学生**（仅看到问题）:  
学生回答内容

**教师**（看到问题+演示）:  
教师行为描述

**训练目标**: 目标描述
{{< /admonition >}}
```

**For actual code, always specify the language:**

```markdown
```python
def example():
    return "code with language specifier"
```
```

**Rules:**
1. Use `{{< admonition example "标题" >}}` for scenario descriptions and examples
2. Use `**加粗**` for labels within admonitions
3. Always specify language for code blocks (e.g., `python`, `bash`, `json`)
4. Never use bare ``` code blocks for non-code content

### Step 4: Reference Existing Posts

For structure and style reference, read existing posts like:
- `/root/workspace/code/blogs/content/posts/rl-razor/index.zh.md`
- `/root/workspace/code/blogs/content/posts/rl-razor/index.en.md`

### Step 5: Generate Output

Create both language versions with:
- Proper frontmatter
- Translated/processed content
- Academic-style citations instead of wiki links
- Consistent formatting with existing posts

## Example

**Input**: `Self-Distillation三部曲-从持续学习到推理优化的范式演进.md`

**Output Directory**: `content/posts/self-distillation-trilogy/`
- `index.zh.md` - Chinese version with wiki links converted to `[1]`, `[2]`, `[3]`
- `index.en.md` - English translation with same citation style

## Citation Conversion Rules

1. Identify all wiki-style links: `[[Paper_Name]]`
2. Assign sequential numbers: `[1]`, `[2]`, `[3]`
3. Replace all occurrences with the assigned number
4. Add a "References" section at the end listing all citations

Example:
```markdown
<!-- Before -->
正如 [[Self-Distillation_Enables_Continual_Learning]] 中所述...

<!-- After -->
正如 [1] 中所述...

## 参考文献
[1] Self-Distillation Enables Continual Learning — Idan Shenfeld 等 (MIT + ETH)
```
