# Multilingual System Design: EN/KR Language Switching

Date: 2026-03-30

## Goal
- Add EN/KR toggle button in header
- Filter posts by selected language
- Support gradual translation of existing posts

## Current Post Language Status (58 posts)

| Language | Count | Categories |
|----------|-------|------------|
| EN | 25 | CS224N(12), Projects(6), ML(4), etc |
| KR | 23 | Computer Science(5), Setup(5), ML(7), Etc(3), Earbuds(2), Math(1) |
| EN+KR mixed | 8 | CS224N/w2(3), Linear Algebra(1), ML(4) |

### Detailed Breakdown

```
EN       | CS224N series (12 posts)
EN       | Projects: phy-interface, delay-locked-loop, olive-pro-max, equalizer, speech-enhancement-tinyml, speech-enhancement-ha
EN       | ML: NAS, TinyML-overview, On-device-learning, Vision-Transformer
EN       | Earbuds: make-product-part1, make-product-part2

KR       | Computer Science: Fixed-point, Binary-tree, Process-Thread, CPU-Scheduling, Preprocess-Compile-Linker
KR       | Setup: all 5 posts (disk-partition, gpu-conda, os-setup, samba, user-group)
KR       | ML: Optimization-tiny-engine 1&2, Quantization 1&2, Pruning 1&2, Smart-pointer
KR       | Etc: 글또를 시작하며, Git-Recipe, CMake-and-Build
KR       | Earbuds: remove-the-noise, platform-for-microphone-streaming
KR       | Math: Taylor-Series
KR       | Device: earbud-project

EN+KR    | CS224N/w2: greedy-deterministic, neural-dependency-parser, stochastic-gradient-descent
EN+KR    | Linear Algebra: eigenvalue-eigenvector
EN+KR    | ML: statistic-basic, Knowledge-Distillation, Transformer-for-TinyML, Various-Applications
```

## Approach Options

### Option A: Front Matter Based (Recommended)

**No GitHub Actions required. Works with current GitHub Pages setup.**

Each post gets `lang` and `ref` front matter:

```yaml
# Existing Korean post
---
title: "Fixed point vs Floating point"
lang: ko
ref: fixed-floating-point
---

# New English translation
---
title: "Fixed point vs Floating point"
lang: en
ref: fixed-floating-point
---
```

Language toggle in header uses Liquid to find matching `ref`:

```liquid
{% assign alt = site.posts | where: "ref", page.ref %}
{% for post in alt %}
  {% if post.lang != page.lang %}
    <a href="{{ post.url }}">{{ post.lang | upcase }}</a>
  {% endif %}
{% endfor %}
```

Post listing filtered by language:

```liquid
{% assign filtered_posts = site.posts | where: "lang", page.lang %}
```

UI strings in `_data/translations.yml`:

```yaml
en:
  nav_about: "About"
  nav_projects: "Projects"
  nav_archive: "Archive"
ko:
  nav_about: "소개"
  nav_projects: "프로젝트"
  nav_archive: "아카이브"
```

**Pros:**
- No plugins, no Actions setup
- Minimal structural change (single _posts/ directory)
- Gradual migration possible

**Cons:**
- _posts/ directory grows 2x as translations added
- Manual Liquid filtering
- No automatic SEO hreflang tags

### Option B: Folder Based

Separate `/en/_posts/` and `/ko/_posts/` directories.

**Pros:** Clean URL structure (`/en/...`, `/ko/...`)
**Cons:** Major restructuring, duplicated layouts

### Option C: Polyglot Plugin (GitHub Actions)

Plugin handles routing, fallback, and sitemap automatically.

**Pros:** Auto fallback for untranslated posts, SEO tags
**Cons:** Requires GitHub Actions workflow setup

## Implementation Plan (Option A)

### Phase 1: Foundation
1. Add `lang` and `ref` to all existing posts (batch script)
2. Create `_data/translations.yml`
3. Add EN/KR toggle to `_includes/header.html`

### Phase 2: Filtering
4. Modify post listing templates to filter by language
5. Store language preference in localStorage
6. Default to EN

### Phase 3: Translation
7. Translate posts incrementally (prioritize by traffic/importance)
8. Toggle only shows when translation exists

## Key Files to Modify

- `_includes/header.html` — toggle button
- `_layouts/home.html` — post listing filter
- `_data/translations.yml` — UI strings (new)
- `_sass/custom.scss` — toggle button styling
- All `_posts/*.md` — add lang/ref front matter

## Notes

- Posts without translation: toggle button hidden or links to same language
- Mixed EN+KR posts: assign primary language, translate later
- about.md, projects.md, history.md also need lang versions
