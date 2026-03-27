# Typography Comparison: blog.samaltman.com vs Current Site

Date: 2026-03-27

## Sam Altman Blog (blog.samaltman.com)

Platform: Posthaven
CSS sources:
- `https://phthemes.s3.amazonaws.com/189/ocI2l2NFgWKLlp1H/blog.css?v=1496776340` (theme)
- `/assets/blog-internal-ebf9cc379e51c299993a0f443d1fee65.css` (platform)

### Body Text
| Property | Value |
|----------|-------|
| font-family | `"Oxygen", sans-serif` (Google Font) |
| font-size | 16px |
| line-height | 1.6 |
| font-weight | 400 |
| color | #555 |
| letter-spacing | (none) |

### Headings (h1-h6)
| Property | Value |
|----------|-------|
| font-family | `"Crimson Text", Georgia, serif` (Google Font) |
| font-weight | 400 (light, not bold) |
| line-height | 1.1 |
| letter-spacing | -0.015em |
| h1 font-size | 50px (general), 30px (responsive), 26px (.post-title) |
| h2 font-size | 28px |
| h3 font-size | 22px |

### Links
| Property | Value |
|----------|-------|
| color | #2b6cc2 |
| text-decoration | none (underline on hover) |

### Paragraph
| Property | Value |
|----------|-------|
| margin | 24px 0 |

### Key Design Choices
- Serif headings + sans-serif body = classic editorial look
- Heading weight 400 = elegant, not heavy
- Letter-spacing -0.015em on headings = tighter, more refined
- Body color #555 = softer than black, easier to read
- Two Google Fonts: Oxygen (body) + Crimson Text (headings)

---

## Current Site (ooshyun.github.io)

Platform: Jekyll with jekyll-text-theme
CSS sources: `_sass/common/_variables.scss`, `_sass/custom.scss`

### Body Text
| Property | Value |
|----------|-------|
| font-family | `-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif` (system stack) |
| font-size | 16px (1rem), 14px on small screens |
| line-height | 1.6 (body), 1.8 (article content) |
| font-weight | 400 |
| color | #222 |
| letter-spacing | (none) |

### Headings (h1-h6)
| Property | Value |
|----------|-------|
| font-family | same system font stack |
| font-weight | 700 (bold) |
| line-height | 1.6 |
| letter-spacing | (none) |
| h1 font-size | 2.5rem (40px) |
| h2 font-size | 1.9rem (30.4px) |
| h3 font-size | 1.5rem (24px) |

### Links
| Property | Value |
|----------|-------|
| color | #416d2e (green) |
| font-weight | 700 |

### Key Design Choices
- System font stack = no external font loading, fast
- Bold headings (700) = strong, utilitarian
- Dark text (#222, #000) = high contrast
- Green link color = theme identity

---

## Differences Summary

| Property | Sam Altman Blog | Current Site | Change Needed |
|----------|----------------|--------------|---------------|
| Body font | Oxygen (Google Font) | System fonts | Import Oxygen |
| Heading font | Crimson Text (serif) | System fonts | Import Crimson Text |
| Heading weight | 400 | 700 | Reduce to 400 |
| Heading letter-spacing | -0.015em | none | Add -0.015em |
| Heading line-height | 1.1 | 1.6 | Reduce to 1.1 |
| Body color | #555 | #222 | Lighten to #555 |
| Heading color | #555 | #000 | Lighten to #555 |
| Link color | #2b6cc2 (blue) | #416d2e (green) | Optional change |
| Paragraph margin | 24px 0 | theme default | Set 24px 0 |

---

## Implementation Plan (if applied)

### Step 1: Import Google Fonts
Add to `_includes/head.html`:
```html
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;700&family=Oxygen:wght@400;700&display=swap" rel="stylesheet">
```

### Step 2: Override Variables
Add to `_sass/custom.scss`:
```scss
// Typography overrides (Sam Altman blog style)
body {
  font-family: "Oxygen", sans-serif;
  color: #555;
}

h1, h2, h3, h4, h5, h6 {
  font-family: "Crimson Text", Georgia, serif;
  font-weight: 400;
  line-height: 1.1;
  letter-spacing: -0.015em;
  color: #555;
}

.article__content p {
  margin: 24px 0;
}
```

### Step 3: Scope Decision
- **Global**: Apply to entire site (all pages)
- **About page only**: Wrap overrides in `.page--about` or similar scope
- **Selective**: Apply to article content only via `.article__content`

### Notes
- Adding Google Fonts increases page load (~50-100ms)
- Crimson Text serif headings significantly change the visual identity
- Consider testing both fonts locally before committing
- The green link color (#416d2e) is a strong theme identity element; changing to blue (#2b6cc2) would alter site branding
