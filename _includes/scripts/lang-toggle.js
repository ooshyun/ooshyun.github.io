/* Language toggle + list filtering, shared by the minimal layout and TeXt post pages.
 * - <html data-lang="en|ko"> drives CSS that shows one language's inline copies.
 * - On a post/page that has a translation (meta page-ref + window.__langData), the toggle navigates to it.
 * - On list pages, .post rows are filtered by language, keyword (#tag=) and the search box. */
(function () {
  var root = document.documentElement;
  var STORAGE_KEY = 'preferred_lang';
  var state = { lang: 'en', tag: '', q: '' };
  var postsList = document.getElementById('posts');
  var LIMIT = postsList ? parseInt(postsList.getAttribute('data-limit') || '0', 10) : 0;
  var searchBox = document.getElementById('search');
  var resultLine = document.getElementById('result');
  var emptyState = document.getElementById('empty');
  var pageRefMeta = document.querySelector('meta[name="page-ref"]');
  var pageLangMeta = document.querySelector('meta[name="page-lang"]');

  function stored() { try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; } }
  function store(lang) { try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {} }

  function findAlt(ref, lang) {
    var d = window.__langData || {};
    var lists = [d.posts || [], d.pages || []];
    for (var k = 0; k < lists.length; k++) {
      for (var i = 0; i < lists[k].length; i++) {
        if (lists[k][i].ref === ref && lists[k][i].lang === lang) return lists[k][i].url;
      }
    }
    return null;
  }

  function matches(li) {
    var langs = (li.getAttribute('data-langs') || '').split(' ');
    if (langs.indexOf(state.lang) === -1) return false;
    if (state.tag && (li.getAttribute('data-tags') || '').split(' ').indexOf(state.tag) === -1) return false;
    if (state.q && (li.getAttribute('data-search') || '').indexOf(state.q) === -1) return false;
    return true;
  }

  function refresh() {
    root.setAttribute('data-lang', state.lang);
    document.querySelectorAll('.lang-toggle__btn').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-set') === state.lang));
    });
    document.querySelectorAll('.tags button').forEach(function (b) {
      b.setAttribute('aria-pressed', String((b.getAttribute('data-tag') || '') === state.tag));
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
  }

  function setLang(lang) {
    store(lang);
    /* reading a translated post/page: jump to the other version */
    if (pageRefMeta) {
      var alt = findAlt(pageRefMeta.getAttribute('content'), lang);
      if (alt && lang !== (pageLangMeta && pageLangMeta.getAttribute('content'))) { location.href = alt; return; }
    }
    state.lang = lang;
    refresh();
  }

  /* initial language: the page's own language when it has one, else stored preference, else browser */
  var initial = pageLangMeta ? pageLangMeta.getAttribute('content') : stored();
  if (initial !== 'ko' && initial !== 'en') {
    initial = (navigator.language || '').toLowerCase().indexOf('ko') === 0 ? 'ko' : 'en';
  }
  state.lang = initial;

  document.querySelectorAll('.lang-toggle__btn').forEach(function (b) {
    b.addEventListener('click', function (e) { e.preventDefault(); setLang(b.getAttribute('data-set')); });
  });
  document.querySelectorAll('.tags button').forEach(function (b) {
    b.addEventListener('click', function () {
      var t = b.getAttribute('data-tag') || '';
      state.tag = (state.tag === t) ? '' : t;
      if (history.replaceState) history.replaceState(null, '', state.tag ? '#tag=' + encodeURIComponent(state.tag) : location.pathname);
      refresh();
    });
  });
  var hashTag = /#tag=([^&]+)/.exec(location.hash);
  if (hashTag) state.tag = decodeURIComponent(hashTag[1]);
  if (searchBox) {
    searchBox.addEventListener('input', function () { state.q = searchBox.value.trim().toLowerCase(); refresh(); });
    if (location.hash === '#search') searchBox.focus();
  }

  refresh();
})();
