# Minimal researcher-site survey (verified 2026-09-12)

Target structure for ooshyun.github.io home page: **Bio (main) → Publications/Projects → Writing → Side projects (unreleased)**.

Method: every URL curl-checked (200 OK); HTML + linked CSS pulled and grepped for real values. "n/m" = not in CSS / browser default.
Dropped as dead/unusable: stellabiderman.com (no DNS), aroyer.github.io (404), nlp.seas.harvard.edu/rush (→ rush-nlp.com), michaelnielsen.org (host placeholder → measured michaelnotebook.com), gollakota.cs.washington.edu (no DNS), yaofu (Notion), neelnanda.io / jasonwei.net (Squarespace).

## 1. Per-site dissection (16 sites)

| # | Site / who | Stack | Home sections (order) | Nav | Pubs format | Writing list | Photo | Borrow |
|---|---|---|---|---|---|---|---|---|
| 1 | [karpathy.ai](https://karpathy.ai) — Karpathy | hand HTML, 4 KB CSS | one page: header → timeline → talks → teaching → writing → pet projects → publications → misc | none | plain blocks `.pub{border-left:4px solid #aaa}`, venue `#090`, no self-bold, no [pdf] | `Mar 2021 <a>Title</a>` date-left inline, no grouping | 240px circle, left of name | left-rail timeline; cheapest pub block |
| 2 | [jonbarron.info](https://jonbarron.info) — Barron | hand HTML tables | one page: bio+photo → Research → Miscellanea → Talks → Service → Teaching | none | 20/80 table rows, 160px thumb, `<strong>` self, `<em>` venue, `project page / arXiv / bibtex` words, `#ffffd0` highlight rows | none | circle, 37% td, right | the pub row; highlight for selected papers |
| 3 | [homes.cs.washington.edu/~jessejm](https://homes.cs.washington.edu/~jessejm/) — UW PhD | Barron clone | bio → Publications → Current Projects → Past Projects | none | `[ PDF ] <papertitle>` + `<b>` self + `<em>` venue in one boxed cell | none | pre-cropped circle, right | `[ PDF ]` prefix; "Current Projects" heading |
| 4 | [tridao.me](https://tridao.me) — Tri Dao | Jekyll al-folio | name → bio+float-right photo → interests → students → awards → **latest posts** → **selected pubs** | top-right fixed: About/Blog/Publications + theme toggle | year-grouped, venue badge col, `.title` bold, self `<em>` underlined, pill buttons Abs/PDF/Code | `table-borderless`: 20% date th + title, no excerpt | 30% rounded rect | home = bio + posts + pubs is exactly the target structure |
| 5 | [rush-nlp.com](https://rush-nlp.com) — Sasha Rush | Jekyll minima + BS3 | photo+name → 18px intro → Bio → paper thumb table → Funding | top-right: Main/Group/Projects/Publications | 150px thumb td + title/authors/venue, no self-bold | none | 200px `img-rounded`, left | bold `Role. Org. City.` tagline |
| 6 | [horace.io](https://horace.io) — Horace He | hand HTML | photo+h1 → 2 ¶ → `[Email] [Resume] [Scholar]` → "Why you might know me" bullets | full-width 5-cell bar | /research: 200px thumb float-left + `[Paper] [Twitter Thread]` | /writing list | square, float right 40% | bracket links; year-stamped one-liners |
| 7 | [lucasb.eyer.be](https://lucasb.eyer.be) — Beyer | hand HTML full-screen sections | hero → Things I've done (Publications/Talks/Projects; Hobby) → Writing → CV | fixed left, collapsible | `<li><a>PaliGemma</a>: one-sentence TL;DR` | plain `<ul>` titles, no dates | none | pubs-as-TL;DR; `<b>2025—now:</b>` open-ended project row |
| 8 | [xeiaso.net](https://xeiaso.net) — Xe Iaso | custom + Tailwind 3.4 | h1+role → avatar intro card → **Recent Articles** → support → **Notable Publications** → **Highlighted Projects** → Quick Links | full-width top bar | `ul.list-disc`: link + `<br>` blurb | `<time class="font-mono">09/12/2026</time>` left + title | ~96px avatar in bordered card | closest existing analogue of Bio/Writing/Pubs/Projects on one page |
| 9 | [thesephist.com](https://thesephist.com) — Linus Lee | Hugo | h1 "My name is Linus." → prose → Speaking → Get in touch | top-left: posts/projects/stream | n/a | /posts: year `h2`, title + date RIGHT (`.dateprefix` italic .8em #778), ✱ for starred | none | tiered /projects: Highlights → Released → Retired → Experiments → Unfinished |
| 10 | [macwright.com](https://macwright.com) — MacWright | Eleventy | `Writing ⇢` → `Micro ⇢` → `Photos ⇢` → `Reading ⇢` | left column ≥1025px | /projects 3-col thumb grid | grid `1fr min-content`: title left, ISO date right, `tabular-nums` | none | heading-as-link "Section ⇢"; tidiest date-right row |
| 11 | [lilianweng.github.io](https://lilianweng.github.io) — Weng | Hugo PaperMod | welcome card → 10 post cards | top-right: Posts/Archive/Search/Tags/FAQ | none | card: h2 → 2-line excerpt → "Date · N min" | none | 720px column; `--primary/--secondary/--border` token trio |
| 12 | [sebastianraschka.com](https://sebastianraschka.com) — Raschka | Jekyll 4.2.1 | intro + 200px circle right + CTA pills → Recent (featured + 120×80 thumbs) | top-left 2-row | none on home | /blog: year `h2` → date above title (0.8em #828282) | 200px circle, right | `[data-theme]` tokens; date-above-title |
| 13 | [eugeneyan.com](https://eugeneyan.com) — Yan | Jekyll + BS4 | "Hi, I'm Eugene" → bullets → newsletter → **Latest** → Talks → Prototypes | top-right pills | none | `<code>21 Jun 2026</code> · Title` | none | `<code>` date list; monospace pill tags |
| 14 | [yoonholee.com](https://yoonholee.com) — Yoonho Lee (Stanford, Korean) | Jekyll, trimmed al-folio | home = photo + bio only | top: About/Blog/Papers/CV | /papers: `<b>` title, `<strong>` self, `<em>` venue, `[abstract][arXiv][code]`, year sidebar | /blog: `h3` title + "14 min read · 2026-06-08" | 200px rounded rect | 720px/17px/underline-offset links; bracket link row |
| 15 | [sanghyukchun.github.io](https://sanghyukchun.github.io/blog/) — Sanghyuk Chun (KO/EN) | Hugo + BS5 | landing card → `/home/` (News → Publications → Activities) and `/blog/` | breadcrumb | numbered `[C36]`, anchored ids | year `h2` → ISO date → title → excerpt → categories incl. `English` | 250px circle | only researcher precedent for a mixed KO/EN list with a language tag |
| 16 | [michaelnotebook.com](https://michaelnotebook.com) — Nielsen | pandoc | h1 → right sidecard → intro → Selected recent work (4-up cards) → Books | none | cards: cover + title + date | tags incl. `ongoing`, `rough` | 27% float-right sidecard | tag-as-status; sidecard |

Also measured but cut: brandur.org (`max-w-[750px]`, `#f6f5e9` warm bg, exemplary `/now` page), jvns.ca, blog.wesleyac.com (date-right flex, `#615b5b`), huyenchip.com (Lora 16/1.5 on `#fdfdfd`), finbarr.ca (13px, too small), pliang (Arial, 960px), natolambert.com (Webflow), malekitani.github.io (al-folio, 800px; Gollakota-group sites are pubs-only, none blog).

## 2. Measured CSS comparison

| Site | max-width | body font / size / lh | text | bg | links | dark |
|---|---|---|---|---|---|---|
| karpathy | `.container{width:970px}` | `sans-serif` / 16px / 1.4 | #333 | white | default | no |
| jonbarron | table 800px | Lato,Verdana / 14px | n/m | n/m | #1772d0, none, hover #f09228 | no |
| jessejm | table 800px | Lato / 14px | n/m | n/m | mediumpurple, none | no |
| tridao | 930px | Roboto 300 / 16px / 1.5 | #000 | #fff | #b509ac, hover underline | toggle `[data-theme]` |
| rush-nlp | 840px | Helvetica Neue / 14px / 1.43 | #333 | #fff | #337ab7 | no |
| horace | `body{max-width:900px}` | Palatino / 18px / 1.6 | #444 | default | rgb(82,119,221), none | no |
| lucasb | cols ≤500px | DejaVu Sans / 0.9em / 1.5 | #585858 | #e8e8e8 | #6bd1fb | media |
| xeiaso | 1024px | Schibsted Grotesk / 16px / 1.5 | rgb(40,40,40) | #f9f5d7 | rgb(184,0,80) underline | media |
| thesephist | 700px | IBM Plex Serif / 18px / 1.625 | #222 | #f8f8f8 | text color + 2px #11b6a5 underline | no |
| macwright | 640px | system sans / 1rem / 1.6 | light-dark(#111,#ccc) | light-dark(#fff,#111) | underline | `light-dark()` + toggle |
| lilianweng | `--main-width:720px` | system sans / 16px / 1.6 | rgb(30,30,30) | #fff | none | auto + class |
| raschka | 740px | Helvetica,Arial / 16px / 1.5 | #111 | #fff | #0479a8 | toggle + system |
| eugeneyan | 750px | Merriweather / 16px / 1.5 | #333 | #fff | #007bff | manual toggle |
| yoonholee | 720px | Source Sans 3, system-ui / 17px / 1.5 | #262626 | #fff | #a21818 underline, offset .22em | no |
| sanghyukchun | 740px | system-ui / 1rem / 1.5 | #5c5c5c | #f8f9fa | #4684e0 | no |
| michaelnotebook | `42em` (≈588px) | Georgia / 14px / 1.6 | #3a3226 | #fff8e7 | #1a4c96, none | no |

Cross-site facts: blog-first sites cluster at **640–750px**; pub-table sites go 800–970px. Body 16–18px, line-height 1.5–1.6 is universal among modern ones. Nobody uses pure black text. 10/16 use `text-decoration:none` + hover underline; the three most typographically careful (thesephist, macwright, yoonholee) underline always with offset/colour tricks. Dark mode: 7/16, always via CSS custom properties. Nav is top-right or absent.

## 3. Three archetypes

**A. "Barron table"** (thumbnail pub rows on one page): jonbarron, jessejm, rush-nlp, horace/research, tridao & al-folio pubs.
Pros: publications are the hero; thumbnails suit systems/mobile work. Cons: needs 800–930px, which makes an 80-post writing list look like a spreadsheet; sections end up unequal; table markup fights mobile. Fit: good for section 2 only.

**B. "Karpathy long scroll"** (bio header, then stacked sections, no nav): karpathy, xeiaso, eugeneyan, lucasb, michaelnotebook.
Pros: literally the 4-section spec on one page; xeiaso.net already does Bio → Recent Articles → Publications → Projects; headings double as anchors. Cons: 80 posts cannot all sit on the home page, so you need "latest 5–8 + `Writing ⇢` link" and a `/writing` archive; long single pages age badly.

**C. "Weng feed with bio header"** (bio card, then chronological post list; pubs on a subpage): lilianweng, raschka, huyenchip, thesephist, yoonholee, sanghyukchun.
Pros: 720px column suits Korean prose; writing stays fresh automatically; cleanest typography. Cons: pubs demoted to a nav item; a thin /papers page for a PhD student.

**Recommendation: B with C's column.** One 720px page: bio → selected pubs (plain rows, no thumbnails) → latest writing (8 rows + `Writing ⇢` to a year-grouped archive) → side projects; nav limited to `Writing · Papers · CV` top-right, so the two long lists get real pages.

## 4. "Side projects (unreleased)" — three patterns

No surveyed academic site uses a styled "coming soon" card or mailing-list teaser; they signal WIP by **tiering, dating, or tagging**.

1. **Open-ended dated one-liner row (recommended).** Lucas Beyer: `<li><b>ca 2025—now:</b> Establish OpenAI's Zürich office…` (no link); stephango.com/projects: `<span class="muted">2024—</span> Project · one-line description`; horace.io: `<span class="date">2020 - Present</span>`. Pair with a teaser clause in the bio (Matuschak "My current focus is…", Lambert "He is currently doing something new").
   Markup: `<li><span class="muted">2026—</span> On-device agent runtime <span class="muted">— in progress</span></li>`
2. **Status-tiered list.** thesephist.com/projects: Highlights → Released → Retired → **Experiments** → **Unfinished**. gwern: `<span class="page-status">in progress</span>`; michaelnotebook: `ongoing`/`rough` tags. Use text words, not coloured pills.
3. **Dated log / Now page.** brandur.org/now, sivers.org/now, maggieappleton.com/now. Shows momentum without a release, but rots if not updated.

Skip: product-style coming-soon + email capture. To hide the section entirely, Chip Huyen's approach is an HTML comment in source; ships with one uncomment.

## 5. Starting spec (values traced to measured sites)

**Layout**
- Content column: `max-width: 720px; padding: 0 24px` (lilianweng, yoonholee, hugo-bearblog). 800px only if thumbnails are required (jonbarron/malekitani).
- Nav: top-right inline text, 3 items max (tridao/yoonholee placement). Or none (karpathy) if all four sections live on one page.
- Section spacing: `section + section { margin-top: 3rem }` (karpathy `.ctitle{margin-top:40px}`); rows `margin: .35rem 0`.
- Photo: 160–200px, `border-radius: 50%`, right of bio text (raschka; barron/jessejm right td). Rounded-rect 200px (yoonholee) if less "avatar".

**Type** (Korean fallback added — none of the Korean-authored sites declares one)
- Body: `font-family: "Source Sans 3", -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", Pretendard, "Noto Sans KR", "Segoe UI", sans-serif; font-size: 17px; line-height: 1.6` (yoonholee 1.0625rem; lilianweng/macwright 1.6).
- Korean bodies: `:lang(ko) { line-height: 1.75; word-break: keep-all }` (kh-kim `line-height:1.8`).
- Optional serif variant (thesephist IBM Plex Serif 18/1.625, eugeneyan Merriweather 16/1.5) paired with `"Noto Serif KR"`; heavier download, sans is the safer default.
- Headings: same family, `h1 1.5rem/700`, `h2 1.125rem/600` small section labels (thesephist, macwright), or karpathy-style lowercase labels.
- Dates/meta: `font-variant-numeric: tabular-nums; color: var(--muted)` (macwright), optionally monospace (eugeneyan, xeiaso, horace `.date{color:#aaa;font-family:Monaco;font-size:80%}`).

**Colour** (tokens so dark mode is one block)
- `--text:#262626` (yoonholee) or `#222` (thesephist); `--muted:#778` (thesephist) / `#828282` (raschka, al-folio); `--bg:#fff` or warm `#f8f8f8` (thesephist) / `#fdfdfd` (huyenchip); `--link:#a21818`-style single accent (yoonholee) or text-coloured links with 2px accent underline (thesephist `text-decoration-color:#11b6a5`); `--rule:#eee`.
- Dark: `@media (prefers-color-scheme: dark)` → `--bg:#1c1c1d; --text:#e8e8e8` (al-folio), or macwright's `light-dark()` for 2024+ browsers.
- Links: `text-decoration: underline; text-underline-offset: .22em; text-decoration-thickness: 1px` (yoonholee). Skip hover-only underline (hurts on touch).

**Publication row** (no thumbnails; barron/yoonholee hybrid)
```html
<li class="pub">
  <span class="pub-title"><a href="…">Aurchestra: …</a></span>
  <span class="pub-authors"><b>Seunghyun Oh</b>, Malek Itani, …</span>
  <span class="pub-venue"><em>MobiSys 2026</em></span>
  <span class="pub-links">[<a>pdf</a>] [<a>code</a>] [<a>site</a>]</span>
</li>
```
`<b>` self (barron/yoonholee), `<em>` venue (barron/tridao), bracket links (horace/yoonholee/jessejm), optional muted year column at 56px. Barron's `#ffffd0` highlight or a `★` gutter for selected papers instead of a separate section.

**Writing row** (macwright grid + Chun's language tag)
```html
<li class="post" lang="ko">
  <a href="…">고정소수점 vs 부동소수점</a>
  <span class="post-meta"><span class="lang">KO</span> <time>2023-02-24</time></span>
</li>
```
`.post{display:grid;grid-template-columns:1fr min-content}` title-left, date-right tabular (macwright; wesleyac). Home shows 8 newest under `h2 > a "Writing ⇢"` (macwright); `/writing` groups by year `h2` (thesephist, raschka, sanghyukchun), excerpt off, `KO`/`EN` text tag. Tags off on home.

**Side projects row:** pattern 1 above, same `li` grid with muted `2026—` date column and no link; bio gets one teaser clause.
