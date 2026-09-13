#!/usr/bin/env python3
"""Build the mockup pages (home / writing / projects) from real post data.

Run from anywhere:  python3 docs/mockups/build.py
Outputs (in docs/mockups): home-v2.html (artifact fragment), index.html (standalone home),
                           writing.html, projects.html, site.css, site.js
"""
import glob
import html
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
TAG_BLOCKLIST = {"lang:", "en", "ko"}


# --------------------------------------------------------------------------- data
def load_posts():
    posts = []
    for p in glob.glob(os.path.join(ROOT, "_posts", "**", "*.md"), recursive=True):
        s = open(p, encoding="utf-8", errors="ignore").read()
        m = re.match(r"---\n(.*?)\n---", s, re.S)
        if not m:
            continue
        fm = m.group(1)

        def g(k):
            mm = re.search(r"^" + k + r":\s*(.*)$", fm, re.M)
            return mm.group(1).strip().strip("\"'") if mm else ""

        title = g("title")
        if not title:
            continue
        date = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(p))
        if not date:
            continue
        lang = g("lang") or "en"
        ref = g("ref") or re.sub(r"^\d{4}-\d{2}-\d{2}-", "", os.path.splitext(os.path.basename(p))[0])
        raw = g("tags") or g("tag")
        tags = [t for t in re.split(r"[\s,]+", raw.strip("[]")) if t and t not in TAG_BLOCKLIST]
        posts.append({"date": date.group(1), "title": title, "lang": lang, "ref": ref, "tags": tags})
    merged = {}
    for post in posts:
        key = post["ref"]
        if key not in merged:
            merged[key] = {"date": post["date"], "titles": {}, "langs": set(), "tags": []}
        merged[key]["titles"][post["lang"]] = post["title"]
        merged[key]["langs"].add(post["lang"])
        merged[key]["date"] = max(merged[key]["date"], post["date"])
        for t in post["tags"]:
            if t not in merged[key]["tags"]:
                merged[key]["tags"].append(t)
    return sorted(merged.values(), key=lambda x: x["date"], reverse=True)


POSTS = load_posts()
TAG_COUNTS = Counter(t for p in POSTS for t in p["tags"])


def bi(en, ko):
    return f'<span lang="en">{en}</span><span lang="ko">{ko}</span>'


def post_li(item):
    langs = " ".join(sorted(item["langs"]))
    t = item["titles"]
    if len(t) == 2 and t.get("en") != t.get("ko"):
        title = f'<span lang="en">{html.escape(t["en"])}</span><span lang="ko">{html.escape(t["ko"])}</span>'
    else:
        title = html.escape(t.get("en") or t.get("ko"))
    search_text = html.escape(" ".join(list(t.values()) + item["tags"]).lower(), quote=True)
    chip = '<span class="chip">EN/KO</span>' if len(item["langs"]) == 2 else ""
    tags = " ".join(item["tags"])
    return (
        f'<li class="post" data-langs="{langs}" data-tags="{html.escape(tags, quote=True)}" data-search="{search_text}">'
        f'<span><a href="#">{title}</a>{chip}</span>'
        f'<time class="mono" datetime="{item["date"]}">{item["date"]}</time></li>'
    )


# --------------------------------------------------------------------------- shared css / js
CSS = r"""
:root {
  --bg: #F8F7F3;
  --surface: #F5EDE0;          /* intro card: warm tint (option C, chosen) */
  --surface-border: #E6D9C4;
  --ink: #1F1D1A;
  --muted: #6F6A62;
  --rule: #E2DFD7;
  --link: #4B2E83;
  --link-hover: #6A4BB0;
  --mark: #C48A45;
  --chip-bg: #ECE9E1;
  --focus: #C48A45;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1A1917; --surface: #262119; --surface-border: #3B3327; --ink: #E8E4DC; --muted: #9C968B;
    --rule: #34322D; --link: #BCA8EC; --link-hover: #D4C6F5; --mark: #D9A15C; --chip-bg: #2C2A26; --focus: #D9A15C;
  }
}
:root[data-theme="dark"] {
  --bg: #1A1917; --surface: #262119; --surface-border: #3B3327; --ink: #E8E4DC; --muted: #9C968B;
  --rule: #34322D; --link: #BCA8EC; --link-hover: #D4C6F5; --mark: #D9A15C; --chip-bg: #2C2A26; --focus: #D9A15C;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: "IBM Plex Sans", "IBM Plex Sans KR", "Apple SD Gothic Neo", Pretendard, "Noto Sans KR", system-ui, sans-serif;
  font-size: 17px; line-height: 1.6; padding-block: 0 4rem; padding-inline: 24px;
}
:lang(ko), [lang="ko"] { line-height: 1.75; word-break: keep-all; }
a { color: var(--link); text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 0.22em; }
a:hover { color: var(--link-hover); }
a:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 2px; }
b, strong { font-weight: 600; }
.mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
.wrap { max-width: 720px; margin: 0 auto; }

/* language switching */
[data-lang="en"] [lang="ko"]:not(.always) { display: none; }
[data-lang="ko"] [lang="en"]:not(.always) { display: none; }
.post.hide { display: none; }
.year-group.empty { display: none; }

/* nav */
.nav { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; padding-block: 1.1rem 0.9rem; border-bottom: 1px solid var(--rule); }
.brand { display: flex; align-items: center; gap: 0.6rem; text-decoration: none; color: var(--ink); font-weight: 600; }
.brand img { width: 34px; height: 34px; object-fit: contain; }
.nav ul { list-style: none; margin: 0; padding: 0; display: flex; align-items: center; gap: 1.25rem; }
.nav ul a { color: var(--ink); text-decoration: none; }
.nav ul a:hover { color: var(--link); text-decoration: underline; text-underline-offset: 0.22em; }
.nav ul a[aria-current="page"] { color: var(--link); text-decoration: underline; text-underline-offset: 0.22em; }
.nav .search-link { display: inline-flex; align-items: center; color: var(--ink); }
.nav .search-link svg { width: 18px; height: 18px; display: block; }
.lang { display: inline-flex; border: 1px solid var(--rule); border-radius: 999px; overflow: hidden; margin-left: 0.25rem; }
.lang button { appearance: none; border: 0; background: transparent; color: var(--muted); font: inherit; font-size: 0.8rem; padding: 0.15rem 0.6rem; cursor: pointer; font-family: "IBM Plex Mono", ui-monospace, monospace; }
.lang button[aria-pressed="true"] { background: var(--ink); color: var(--bg); }

/* page head */
.head { padding-block: 2.5rem 1.5rem; }
h1 { font-size: 1.85rem; line-height: 1.2; font-weight: 600; margin: 0 0 0.35rem; letter-spacing: -0.01em; text-wrap: balance; }
.role, .lede { margin: 0; color: var(--muted); font-size: 1.05rem; }

/* intro card */
.intro { display: flex; gap: 1.25rem; align-items: flex-start; border: 1px solid var(--surface-border); background: var(--surface); border-radius: 8px; padding: 1.1rem 1.25rem; }
.intro img { width: 96px; height: 96px; flex: 0 0 auto; object-fit: contain; }
.intro p { margin: 0 0 0.75rem; }
.intro p:last-child { margin-bottom: 0; }
.links { display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; font-size: 0.95rem; }
.updated { display: block; text-align: right; color: var(--muted); font-size: 0.8rem; margin-top: 0.75rem; }

/* background block under the card: education + experience */
.background { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem 2rem; margin-top: 1.5rem; }
.background h3 { margin: 0 0 0.5rem; }
.background ul.rows { gap: 0.6rem; }
.background .row { grid-template-columns: 4.2rem 1fr; }
.background .title { font-weight: 500; font-size: 0.98rem; }
.background .meta { font-size: 0.88rem; }

/* sections */
section { margin-top: 3rem; }
.head + section { margin-top: 0; }
.h2row { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.35rem; margin-bottom: 1rem; }
h2 { font-size: 1.15rem; font-weight: 600; margin: 0; letter-spacing: -0.005em; }
.h2row a { font-size: 0.9rem; color: var(--muted); }
h3 { font-size: 0.95rem; font-weight: 600; color: var(--muted); margin: 1.5rem 0 0.6rem; }
h3:first-of-type { margin-top: 0; }
ul.rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.85rem; }

/* year | body rows */
.row { display: grid; grid-template-columns: 5.5rem 1fr; gap: 0 1rem; }
.row .when { color: var(--muted); font-size: 0.9rem; padding-top: 0.15rem; white-space: nowrap; }
.row .title { font-weight: 600; }
.row .meta { color: var(--muted); font-size: 0.95rem; }
.row .meta b { color: var(--ink); }
.row .refs { font-size: 0.9rem; }
.award { color: var(--mark); font-weight: 500; }

/* writing rows */
.post { display: grid; grid-template-columns: 1fr max-content; gap: 0 1rem; align-items: baseline; }
.post time { color: var(--muted); font-size: 0.9rem; }
.chip { display: inline-block; font-size: 0.7rem; padding: 0.05rem 0.4rem; border-radius: 999px; background: var(--chip-bg); color: var(--muted); vertical-align: 0.15em; margin-left: 0.35rem; font-family: "IBM Plex Mono", ui-monospace, monospace; }
.year-group { margin-top: 1.75rem; }
.year-group:first-child { margin-top: 0; }
.year-group h3 { margin: 0 0 0.6rem; }
.note { color: var(--muted); font-size: 0.95rem; margin: 0.75rem 0 0; }

/* writing page: search + tag filter */
.search { position: relative; margin: 0 0 1rem; }
.search svg { position: absolute; left: 0.75rem; top: 50%; transform: translateY(-50%); width: 16px; height: 16px; color: var(--muted); pointer-events: none; }
.search input {
  width: 100%; font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--surface-border); border-radius: 8px;
  padding: 0.55rem 0.75rem 0.55rem 2.3rem;
}
.search input::placeholder { color: var(--muted); }
.tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0 0 1.75rem; padding: 0; list-style: none; }
.tags button {
  appearance: none; font: inherit; font-size: 0.85rem; cursor: pointer; color: var(--ink);
  background: transparent; border: 1px solid var(--rule); border-radius: 999px; padding: 0.1rem 0.65rem;
}
.tags button .n { color: var(--muted); font-size: 0.75rem; margin-left: 0.3rem; font-family: "IBM Plex Mono", ui-monospace, monospace; }
.tags button[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.tags button[aria-pressed="true"] .n { color: var(--bg); opacity: 0.7; }
.result { color: var(--muted); font-size: 0.9rem; margin: 0 0 1rem; }
.empty-state { color: var(--muted); margin: 1rem 0; display: none; }
.empty-state.show { display: block; }

footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--rule); color: var(--muted); font-size: 0.9rem; display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
footer a { color: var(--muted); }


@media (max-width: 560px) {
  body { font-size: 16px; }
  .intro { flex-direction: column; }
  .intro img { width: 80px; height: 80px; }
  .background { grid-template-columns: 1fr; }
  .row { grid-template-columns: 1fr; }
  .row .when { padding-top: 0; }
  .post { grid-template-columns: 1fr; }
  .nav ul { gap: 0.9rem; }
}
@media (prefers-reduced-motion: no-preference) {
  .lang button, .tags button { transition: background 120ms ease, color 120ms ease; }
}
"""

JS = r"""
(function () {
  var root = document.documentElement;
  var LIMIT = document.getElementById('posts') ? 8 : 0;   /* home shows the 8 latest per language */
  var state = { lang: 'en', tag: '', q: '' };
  var searchBox = document.getElementById('search');
  var resultLine = document.getElementById('result');
  var emptyState = document.getElementById('empty');

  function matches(li) {
    var langs = (li.dataset.langs || '').split(' ');
    if (langs.indexOf(state.lang) === -1) return false;
    if (state.tag && (li.dataset.tags || '').split(' ').indexOf(state.tag) === -1) return false;
    if (state.q && (li.dataset.search || '').indexOf(state.q) === -1) return false;
    return true;
  }

  function refresh() {
    root.setAttribute('data-lang', state.lang);
    root.setAttribute('lang', state.lang);
    document.querySelectorAll('.lang button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.set === state.lang));
    });
    document.querySelectorAll('.tags button').forEach(function (b) {
      b.setAttribute('aria-pressed', String((b.dataset.tag || '') === state.tag));
    });
    var shown = 0;
    document.querySelectorAll('.post').forEach(function (li) {
      var ok = matches(li);
      if (ok) shown++;
      li.classList.toggle('hide', !ok || (LIMIT > 0 && shown > LIMIT));
    });
    document.querySelectorAll('.year-group').forEach(function (g) {
      var any = Array.prototype.some.call(g.querySelectorAll('.post'), function (li) { return !li.classList.contains('hide'); });
      g.classList.toggle('empty', !any);
    });
    if (resultLine) {
      var n = document.querySelectorAll('.post:not(.hide)').length;
      resultLine.textContent = state.lang === 'ko' ? ('글 ' + n + '개') : (n + (n === 1 ? ' post' : ' posts'));
    }
    if (emptyState) emptyState.classList.toggle('show', shown === 0);
    try { localStorage.setItem('preferred_lang', state.lang); } catch (e) {}
  }

  /* language: stored preference, else browser language */
  var initial = null;
  try { initial = localStorage.getItem('preferred_lang'); } catch (e) {}
  if (initial !== 'ko' && initial !== 'en') {
    initial = (navigator.language || '').toLowerCase().indexOf('ko') === 0 ? 'ko' : 'en';
  }
  state.lang = initial;
  document.querySelectorAll('.lang button').forEach(function (b) {
    b.addEventListener('click', function () { state.lang = b.dataset.set; refresh(); });
  });

  /* tag filter (writing page) */
  document.querySelectorAll('.tags button').forEach(function (b) {
    b.addEventListener('click', function () {
      var t = b.dataset.tag || '';
      state.tag = (state.tag === t) ? '' : t;
      if (history.replaceState) history.replaceState(null, '', state.tag ? '#tag=' + encodeURIComponent(state.tag) : location.pathname);
      refresh();
    });
  });
  var hashTag = /#tag=([^&]+)/.exec(location.hash);
  if (hashTag) state.tag = decodeURIComponent(hashTag[1]);

  /* search (writing page) */
  if (searchBox) {
    searchBox.addEventListener('input', function () { state.q = searchBox.value.trim().toLowerCase(); refresh(); });
    if (location.hash === '#search') searchBox.focus();
  }

  refresh();

})();
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400'
    '&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">\n'
    '<link rel="stylesheet" href="site.css">\n'
)

MAGNIFIER = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="11" cy="11" r="7"></circle><line x1="16.5" y1="16.5" x2="21" y2="21"></line></svg>'
)


def nav(current):
    items = [
        ("index.html", "home", bi("About", "소개")),
        ("writing.html", "writing", bi("Writing", "글")),
        ("projects.html", "projects", bi("Projects", "프로젝트")),
        ("#", "cv", bi("CV", "이력서")),
    ]
    lis = []
    for href, key, label in items:
        cur = ' aria-current="page"' if key == current else ""
        lis.append(f'<li><a href="{href}"{cur}>{label}</a></li>')
    lis.append(f'<li><a class="search-link" href="writing.html#search" title="Search posts" aria-label="Search posts">{MAGNIFIER}</a></li>')
    lis.append(
        '<li><div class="lang" role="group" aria-label="Language">'
        '<button type="button" id="lang-en" data-set="en" aria-pressed="true">EN</button>'
        '<button type="button" id="lang-ko" data-set="ko" aria-pressed="false">KO</button>'
        "</div></li>"
    )
    return (
        '<nav class="nav" aria-label="Primary">\n'
        '  <a class="brand" href="index.html"><img src="ooshyun-character.png" alt="" width="34" height="34"><span>Seunghyun Oh</span></a>\n'
        "  <ul>" + "".join(lis) + "</ul>\n</nav>\n"
    )


FOOTER = (
    "<footer>\n"
    "  <span>© 2018–2026 Seunghyun Oh. " + bi("Text licensed CC BY-NC 4.0.", "글은 CC BY-NC 4.0 라이선스를 따릅니다.") + "</span>\n"
    '  <span><a href="mailto:seunghyun.daniel.oh@gmail.com">Email</a> &nbsp; <a href="https://github.com/ooshyun">GitHub</a> &nbsp; '
    '<a href="https://www.linkedin.com/in/seunghyun-oh-106815174/">LinkedIn</a></span>\n'
    "</footer>\n"
)


def page(title, body, current, fragment=False):
    head = f"<title>{title}</title>\n{FONTS}"
    inner = f'<div class="wrap" id="top">\n{nav(current)}{body}{FOOTER}</div>\n<script src="site.js"></script>\n'
    if fragment:
        return head + inner
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n%s</head>\n<body>\n%s</body>\n</html>\n'
        % (head, inner)
    )


# --------------------------------------------------------------------------- content
PAPER = f"""
<li class="row">
  <span class="when mono">2026</span>
  <div>
    <div class="title"><a href="https://doi.org/10.1145/3745756.3809210">Aurchestra: Fine-grained Soundscape Control for Augmented Hearing</a></div>
    <div class="meta"><b>Seunghyun Oh</b>, Malek Itani, Shyamnath Gollakota. <em>MobiSys 2026</em>. <span class="award">{bi("Best Artifact Award", "최우수 아티팩트상")}</span></div>
    <div class="refs">[<a href="https://doi.org/10.1145/3745756.3809210">paper</a>] [<a href="https://github.com/ooshyun/fine_grained_soundscape_control">code</a>] [<a href="/aurchestra/">site</a>]</div>
  </div>
</li>"""

PROJECTS = [
    ("2023", "Speech enhancement with LSTM for TinyML", "엣지 디바이스를 위한 머신러닝 기반 음성 향상",
     "Streaming denoiser small enough for microcontrollers.", "마이크로컨트롤러에서 돌아가는 스트리밍 잡음 제거기.",
     [("code", "https://github.com/ooshyun/Speech-Enhancement-TF"), ("post", "#")]),
    ("2023", "Clarity Challenge: speech enhancement for hearing aids", "보청기를 위한 Clarity Challenge 음성 향상",
     "", "", [("code", "https://github.com/ooshyun/ClarityChallenge2023"), ("post", "#")]),
    ("2022–23", "Olive Max, true wireless earbuds as a hearing aid for severe hearing loss", "Olive Max, 고도 난청을 위한 보청기형 무선 이어버드",
     "", "", [("post", "#")]),
    ("2021", "Equalizer design with cascade and parallel biquad filters", "디지털 필터를 이용한 이퀄라이저 설계",
     "", "", [("code", "https://github.com/ooshyun/FilterDesign"), ("post", "#")]),
    ("2020–21", "Olive Pro, true wireless earbuds for hearing aid", "Olive Pro, 보청기용 무선 이어버드", "", "", [("post", "#")]),
    ("2018–19", "Delay-locked loop for the PHY interface between DRAM and CPU", "DRAM과 CPU 간 PHY 인터페이스를 위한 지연 고정 루프", "", "", [("post", "#")]),
    ("2018–19", "PHY interface between DDR3 and LPDDR3", "DDR3와 LPDDR3 간 PHY 인터페이스", "", "", [("post", "#")]),
]


def project_li(p):
    when, en, ko, den, dko, refs = p
    meta = f'<div class="meta">{bi(den, dko)}</div>' if den else ""
    links = " ".join(f'[<a href="{u}">{bi("post", "글") if k == "post" else k}</a>]' for k, u in refs)
    return (
        f'<li class="row"><span class="when mono">{when}</span><div>'
        f'<div class="title"><a href="#">{bi(en, ko)}</a></div>{meta}<div class="refs">{links}</div></div></li>'
    )


# Bio text copied verbatim from about.md / about-ko.md
BIO_FULL = """
<p lang="en">Hi! I'm a CS PhD student at the <a href="https://www.cs.washington.edu/">University of Washington Paul G. Allen School of Computer Science &amp; Engineering</a>, advised by Prof. <a href="https://homes.cs.washington.edu/~gshyam/">Shyamnath Gollakota</a>. I research proactive AI agents and efficient inference. Previously I built wearable AI systems for augmented hearing; currently I work on proactive agents that interact through speech.</p>
<p lang="en">Before the PhD, I spent 5+ years shipping on-device ML and real-time DSP in audio and wearable devices: cross-platform Sound AI SDK with hardware acceleration (TensorRT/QNN/SNPE) for several target platforms (Syntiant, Arm Ethos, ESP32, Jetson, Edge TPU) at <a href="https://www.cochl.ai">Cochl</a>, streaming speech enhancement on STM32 as a freelance engineer, and hearing aid DSP on Tensilica cores at <a href="https://oliveunion.shop">Olive Union</a>.</p>
<p lang="en">In my free time, I love trail running and reading autobiographies and philosophical essays, especially Walter Isaacson, Albert Camus, and Friedrich Nietzsche.</p>
<p lang="ko">안녕하세요! 저는 <a href="https://www.cs.washington.edu/">워싱턴 대학교 Paul G. Allen School of Computer Science &amp; Engineering</a>에서 <a href="https://homes.cs.washington.edu/~gshyam/">Shyamnath Gollakota</a> 교수님의 지도 아래 연구하고 있는 컴퓨터 과학 박사과정 학생입니다. 능동적 AI 에이전트(Proactive AI Agents)와 효율적 추론(Efficient Inference)을 연구하고 있습니다. 이전에는 증강 청각(Augmented Hearing)을 위한 웨어러블 AI 시스템을 개발했으며, 현재는 음성을 통해 상호작용하는 능동적 에이전트를 연구하고 있습니다.</p>
<p lang="ko">박사과정 이전에는 5년 이상 오디오 및 웨어러블 디바이스 분야에서 온디바이스 ML과 실시간 DSP를 개발했습니다: <a href="https://www.cochl.ai">Cochl</a>에서 여러 타깃 플랫폼(Syntiant, Arm Ethos, ESP32, Jetson, Edge TPU)을 위한 하드웨어 가속(TensorRT/QNN/SNPE) 기반 크로스 플랫폼 Sound AI SDK 개발, 프리랜서로 STM32 기반 스트리밍 음성 향상 시스템 설계, <a href="https://oliveunion.shop">Olive Union</a>에서 Tensilica DSP 코어 기반 보청기 DSP 알고리즘을 개발했습니다.</p>
<p lang="ko">여가 시간에는 트레일 러닝을 즐기고, 자서전과 철학 에세이를 읽는 것을 좋아합니다. 특히 월터 아이작슨, 알베르 카뮈, 프리드리히 니체의 작품을 좋아합니다.</p>
"""

LINKS = (
    '<p class="links"><a href="mailto:seunghyun.daniel.oh@gmail.com">Email</a> <a href="https://github.com/ooshyun">GitHub</a> '
    '<a href="https://www.linkedin.com/in/seunghyun-oh-106815174/">LinkedIn</a> <a href="#">' + bi("CV (PDF)", "이력서 (PDF)") + "</a></p>\n"
    '<span class="updated">' + bi("Website last updated 03/2026", "웹사이트 최종 업데이트 2026년 3월") + "</span>\n"
)

EDU = [
    ("2025–", "University of Washington", "University of Washington", "PhD in Computer Science &amp; Engineering", "컴퓨터 과학 및 공학 박사과정",
     'Advisor. <a href="https://homes.cs.washington.edu/~gshyam/">Shyamnath Gollakota</a>', '지도교수. <a href="https://homes.cs.washington.edu/~gshyam/">Shyamnath Gollakota</a>'),
    ("2021", "Coursera", "Coursera", "DeepLearning.AI TensorFlow Developer", "DeepLearning.AI TensorFlow Developer", "", ""),
    ("2020", "Hanyang University", "한양대학교", "Master in Electronics and Computer Engineering", "전자컴퓨터공학 석사",
     'Advisor. <a href="https://scholar.google.co.kr/citations?hl=en&amp;user=N8CSltQAAAAJ">Changsik Yoo</a>', '지도교수. <a href="https://scholar.google.co.kr/citations?hl=en&amp;user=N8CSltQAAAAJ">유창식</a>'),
    ("2018", "Inha University", "인하대학교", "B.A. in Information and Communication Engineering", "정보통신공학 학사",
     'Advisors. <a href="https://scholar.google.co.kr/citations?user=wcpWpdQAAAAJ">Kichang Kim</a> and <a href="https://scholar.google.com/citations?user=aDmqYYQAAAAJ">Gyungsu Byun</a>',
     '지도교수. <a href="https://scholar.google.co.kr/citations?user=wcpWpdQAAAAJ">김기창</a>, <a href="https://scholar.google.com/citations?user=aDmqYYQAAAAJ">변경수</a>'),
]
EXP = [
    ("2023–25", '<a href="https://www.cochl.ai">Cochl</a>', "Backend engineer for SDK", "SDK 백엔드 엔지니어"),
    ("2023", bi("Freelance", "프리랜서"), "Embedded AI engineer", "임베디드 AI 엔지니어"),
    ("2020–23", '<a href="https://oliveunion.shop">OliveUnion</a>', "Embedded Digital signal processing engineer", "임베디드 디지털 신호처리 엔지니어"),
]


def edu_li(e):
    when, en, ko, den, dko, aen, ako = e
    adv = f'<div class="meta">{bi(aen, ako)}</div>' if aen else ""
    return (
        f'<li class="row"><span class="when mono">{when}</span><div><div class="title">{bi(en, ko)}</div>'
        f'<div class="meta">{bi(den, dko)}</div>{adv}</div></li>'
    )


def exp_li(e):
    when, org, ren, rko = e
    return f'<li class="row"><span class="when mono">{when}</span><div><div class="title">{org}</div><div class="meta">{bi(ren, rko)}</div></div></li>'


# --------------------------------------------------------------------------- home
home_body = f"""
<header class="head">
  <h1>Seunghyun (Conan) Oh</h1>
  <p class="role">{bi("CS PhD @ UW | Sound, Proactive Voice AI Assistant in Conversation, Efficient Inference", "CS PhD @ UW | Sound, Proactive Voice AI Assistant in Conversation, Efficient Inference")}</p>
</header>
<div class="intro">
  <img src="ooshyun-character.png" alt="Poodle character, Seunghyun's profile mark" width="96" height="96">
  <div>
{BIO_FULL}{LINKS}  </div>
</div>
<div class="background">
  <div>
    <h3>{bi("Education", "학력")}</h3>
    <ul class="rows">{"".join(edu_li(e) for e in EDU)}</ul>
  </div>
  <div>
    <h3>{bi("Experience", "경력")}</h3>
    <ul class="rows">{"".join(exp_li(e) for e in EXP)}</ul>
  </div>
</div>

<section id="work">
  <div class="h2row"><h2>{bi("Publications and projects", "논문과 프로젝트")}</h2><a href="projects.html">{bi("All projects", "전체 프로젝트")}</a></div>
  <h3>{bi("Papers", "논문")}</h3>
  <ul class="rows">{PAPER}
  </ul>
  <h3>{bi("Projects", "프로젝트")}</h3>
  <ul class="rows">
    {"".join(project_li(p) for p in PROJECTS[:4])}
  </ul>
</section>

<section id="writing">
  <div class="h2row"><h2>{bi("Writing", "글")}</h2><a href="writing.html">{bi("All posts", "전체 글")}</a></div>
  <ul class="rows" id="posts">
    {"".join(post_li(p) for p in POSTS[:30])}
  </ul>
  <p class="note">{bi("The 8 latest posts written in English. Posts available in both languages appear in both lists.",
                       "한국어로 쓴 최근 글 8개입니다. 두 언어로 쓴 글은 양쪽 목록에 모두 보입니다.")}</p>
</section>

<section id="side">
  <div class="h2row"><h2>{bi("Side projects", "사이드 프로젝트")}</h2></div>
  <ul class="rows">
    <li class="row"><span class="when mono">2026—</span><div>
      <div class="title">{bi("A new side project, in progress", "새 사이드 프로젝트, 진행 중")}</div>
      <div class="meta">{bi("Details here when it is ready to open.", "공개 준비가 되면 여기에 소개합니다.")}</div>
    </div></li>
  </ul>
</section>
"""

# --------------------------------------------------------------------------- writing
by_year = defaultdict(list)
for p in POSTS:
    by_year[p["date"][:4]].append(p)
groups = [
    f'<div class="year-group"><h3 class="mono">{y}</h3><ul class="rows">{"".join(post_li(p) for p in by_year[y])}</ul></div>'
    for y in sorted(by_year, reverse=True)
]
tag_buttons = [f'<li><button type="button" data-tag="" aria-pressed="true">{bi("All", "전체")}</button></li>'] + [
    f'<li><button type="button" data-tag="{html.escape(t, quote=True)}" aria-pressed="false">{html.escape(t)}<span class="n">{n}</span></button></li>'
    for t, n in TAG_COUNTS.most_common()
]
n_en = sum(1 for p in POSTS if "en" in p["langs"])
n_ko = sum(1 for p in POSTS if "ko" in p["langs"])
writing_body = f"""
<header class="head">
  <h1>{bi("Writing", "글")}</h1>
  <p class="lede">{bi(f"{n_en} posts in English. Switch to KO for the {n_ko} posts in Korean.", f"한국어 글 {n_ko}개. EN으로 바꾸면 영어 글 {n_en}개를 볼 수 있습니다.")}</p>
</header>
<section>
  <div class="search">{MAGNIFIER}<input type="search" id="search" placeholder="Search titles and keywords" autocomplete="off"></div>
  <ul class="tags" aria-label="Filter by keyword">{"".join(tag_buttons)}</ul>
  <p class="result" id="result"></p>
  <p class="empty-state" id="empty">{bi("No posts match. Clear the keyword or try another search.", "일치하는 글이 없습니다. 키워드를 해제하거나 다른 검색어를 입력해 보세요.")}</p>
  {"".join(groups)}
</section>
"""

# --------------------------------------------------------------------------- projects
projects_body = f"""
<header class="head">
  <h1>{bi("Publications and projects", "논문과 프로젝트")}</h1>
</header>
<section>
  <div class="h2row"><h2>{bi("Papers", "논문")}</h2></div>
  <ul class="rows">{PAPER}
  </ul>
</section>
<section>
  <div class="h2row"><h2>{bi("Projects", "프로젝트")}</h2></div>
  <ul class="rows">{"".join(project_li(p) for p in PROJECTS)}</ul>
</section>
"""

# --------------------------------------------------------------------------- write
def w(name, content):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(content)


w("site.css", CSS.strip() + "\n")
w("site.js", JS.strip() + "\n")
w("home-v2.html", page("ooshyun.github.io v2", home_body, "home", fragment=True))
w("index.html", page("ooshyun.github.io v2", home_body, "home"))
w("writing.html", page("Writing", writing_body, "writing"))
w("projects.html", page("Projects", projects_body, "projects"))
print(f"posts merged: {len(POSTS)} (en {n_en}, ko {n_ko}); tags: {dict(TAG_COUNTS)}")
