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
