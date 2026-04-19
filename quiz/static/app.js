let editor = null;
let tutorialViewer = null;
let playgroundEditor = null;
let questions = [];
let currentIndex = 0;
let hintCounts = {};
let lastRunPassed = false;
let lastRunQuestionId = null;
const PROGRESS_STORAGE_KEY = "quiz-progress-v1";

function loadProgressStore() {
  try {
    return JSON.parse(window.localStorage.getItem(PROGRESS_STORAGE_KEY) || "{}");
  } catch (error) {
    return {};
  }
}

function saveProgressStore(store) {
  window.localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(store));
}

function getChapterProgress(chapterId, questionCount = 0) {
  const store = loadProgressStore();
  const progress = store[chapterId] || {};
  const completedQuestions = Array.isArray(progress.completed_questions) ? progress.completed_questions : [];
  const answers = progress.answers && typeof progress.answers === "object" ? progress.answers : {};
  const hintCounts = progress.hint_counts && typeof progress.hint_counts === "object" ? progress.hint_counts : {};
  const currentQuestionIndex = Number.isInteger(progress.current_question_index) ? progress.current_question_index : 0;
  const lessonCollapsed = Boolean(progress.lesson_collapsed);
  const completed = Boolean(progress.completed) || (questionCount > 0 && completedQuestions.length >= questionCount);
  return {
    completed,
    completed_questions: completedQuestions,
    answers,
    hint_counts: hintCounts,
    current_question_index: currentQuestionIndex,
    lesson_collapsed: lessonCollapsed,
  };
}

function updateChapterProgress(chapterId, updater, questionCount = 0) {
  const store = loadProgressStore();
  const current = getChapterProgress(chapterId, questionCount);
  const next = updater(current);
  store[chapterId] = {
    completed: Boolean(next.completed),
    completed_questions: Array.from(new Set(next.completed_questions || [])),
    answers: next.answers || {},
    hint_counts: next.hint_counts || {},
    current_question_index: Number.isInteger(next.current_question_index) ? next.current_question_index : 0,
  };
  saveProgressStore(store);
  refreshSavedProgressUi();
}

function describeChapterProgress(progress, questionCount) {
  const completeCount = Math.min(progress.completed_questions.length, questionCount);
  if (progress.completed) {
    return { label: "Completed", progressText: `${questionCount} / ${questionCount} complete`, className: "is-complete" };
  }
  if (completeCount > 0 || Object.keys(progress.answers).length > 0) {
    return { label: "In progress", progressText: `${completeCount} / ${questionCount} complete`, className: "is-in-progress" };
  }
  return { label: "Not started", progressText: `0 / ${questionCount} complete`, className: "" };
}

function applyStatusBadge(element, label, className) {
  if (!element) {
    return;
  }
  element.textContent = label;
  element.classList.remove("is-in-progress", "is-complete");
  if (className) {
    element.classList.add(className);
  }
}

function refreshHomeProgressUi() {
  document.querySelectorAll("[data-chapter-card]").forEach((card) => {
    const chapterId = card.dataset.chapterId;
    const questionCount = Number(card.dataset.questionCount || "0");
    const progress = getChapterProgress(chapterId, questionCount);
    const view = describeChapterProgress(progress, questionCount);
    applyStatusBadge(card.querySelector("[data-chapter-status]"), view.label, view.className);
    const progressText = card.querySelector("[data-chapter-progress-text]");
    if (progressText) {
      progressText.textContent = view.progressText;
    }
  });
}

function refreshSidebarProgressUi() {
  document.querySelectorAll("[data-sidebar-chapter]").forEach((entry) => {
    const chapterId = entry.dataset.chapterId;
    const questionCount = Number(entry.dataset.questionCount || "0");
    const progress = getChapterProgress(chapterId, questionCount);
    const view = describeChapterProgress(progress, questionCount);
    const status = entry.querySelector("[data-sidebar-status]");
    if (status) {
      status.textContent = view.label;
    }
  });

  const currentShell = document.querySelector("[data-current-chapter-status]");
  if (!currentShell) {
    return;
  }

  const chapterId = currentShell.dataset.chapterId;
  const questionCount = Number(currentShell.dataset.questionCount || "0");
  const progress = getChapterProgress(chapterId, questionCount);
  const view = describeChapterProgress(progress, questionCount);
  applyStatusBadge(currentShell.querySelector("[data-current-chapter-badge]"), view.label, view.className);
  const progressText = currentShell.querySelector("[data-current-chapter-progress]");
  if (progressText) {
    progressText.textContent = view.progressText;
  }
}

function refreshSavedProgressUi() {
  refreshHomeProgressUi();
  refreshSidebarProgressUi();
  refreshQuestionTrackerUi();
}

function questionStateFor(question, progress, index) {
  const completed = (progress.completed_questions || []).includes(question.id);
  const hasDraft = Boolean(progress.answers && Object.prototype.hasOwnProperty.call(progress.answers, question.id) && progress.answers[question.id] !== question.starter_code);
  if (completed) {
    return { label: "Completed", className: "is-completed" };
  }
  if (index === currentIndex) {
    return { label: hasDraft ? "Current draft" : "Current", className: "is-current" };
  }
  if (hasDraft) {
    return { label: "Draft", className: "is-draft" };
  }
  return { label: "Not started", className: "" };
}

function syncInlineNextVisibility() {
  showInlineNext();
}

function currentHintCount() {
  const chapter = window.__CHAPTER_DATA__;
  const question = currentQuestion();
  if (!chapter || !question) {
    return 0;
  }
  const progress = getChapterProgress(chapter.id, questions.length);
  return Number(progress.hint_counts[question.id] || 0);
}

function isAnswerUnlocked(question) {
  return Boolean(question && question.answer_code && currentHintCount() >= 3);
}

function syncAnswerTab() {
  const tab = document.getElementById("answer-tab");
  const card = document.getElementById("answer-card");
  const output = document.getElementById("answer-output");
  const question = currentQuestion();
  if (!tab || !card || !output) {
    return;
  }

  card.classList.add("is-hidden");
  output.textContent = "";

  if (isAnswerUnlocked(question)) {
    tab.classList.remove("is-hidden");
  } else {
    tab.classList.add("is-hidden");
  }
}

function revealAnswer() {
  const question = currentQuestion();
  const card = document.getElementById("answer-card");
  const output = document.getElementById("answer-output");
  if (!card || !output || !isAnswerUnlocked(question)) {
    return;
  }
  output.textContent = question.answer_code || "No official answer is available for this question.";
  card.classList.remove("is-hidden");
}

function recordHintUsage() {
  const chapter = window.__CHAPTER_DATA__;
  const question = currentQuestion();
  if (!chapter || !question) {
    return;
  }
  updateChapterProgress(chapter.id, (progress) => ({
    ...progress,
    hint_counts: {
      ...progress.hint_counts,
      [question.id]: Number(progress.hint_counts[question.id] || 0) + 1,
    },
  }), questions.length);
}

function refreshQuestionTrackerUi() {
  if (!window.__CHAPTER_DATA__) {
    return;
  }
  const progress = getChapterProgress(window.__CHAPTER_DATA__.id, questions.length);
  document.querySelectorAll("[data-question-jump]").forEach((item, index) => {
    const question = questions[index];
    if (!question) {
      return;
    }
    const view = questionStateFor(question, progress, index);
    item.classList.remove("is-current", "is-completed", "is-draft");
    if (view.className) {
      item.classList.add(view.className);
    }
    const state = item.querySelector(".question-tracker-state");
    if (state) {
      state.textContent = view.label;
    }
  });
}

function chapterUi() {
  return (window.__CHAPTER_DATA__ && window.__CHAPTER_DATA__.ui) || {};
}

function findActionElements(action) {
  return Array.from(document.querySelectorAll(`[data-action="${action}"]`));
}

function triggerRun() {
  return sendCode("run");
}

function triggerSubmit() {
  return sendCode("submit");
}

function triggerPrimaryShortcut() {
  const question = currentQuestion();
  if (question && lastRunPassed && lastRunQuestionId === question.id) {
    return triggerSubmit();
  }
  return triggerRun();
}

function updateNavigationState() {
  const prevButton = document.getElementById("prev-question");
  const nextButtons = findActionElements("next-question");
  const nextInlineButton = document.getElementById("next-question-inline");
  const progress = document.getElementById("question-progress");
  const question = currentQuestion();

  if (!prevButton || !nextInlineButton || !progress || !question) {
    return;
  }

  prevButton.disabled = currentIndex === 0;
  nextButtons.forEach((button) => { button.disabled = questions.length === 0; });
  nextInlineButton.disabled = questions.length === 0;
  progress.textContent = `${currentIndex + 1} / ${questions.length}`;
  prevButton.textContent = "Previous";
  const nextLabel = currentIndex === questions.length - 1
    ? "Finish Chapter"
    : "Next";
  nextButtons.forEach((button) => { button.textContent = nextLabel; });
}

function hideInlineNext() {
  const button = document.getElementById("next-question-inline");
  if (button) {
    button.classList.add("is-hidden");
  }
}

function showInlineNext() {
  const button = document.getElementById("next-question-inline");
  if (button) {
    button.classList.remove("is-hidden");
  }
}

function goToNextQuestion() {
  if (currentIndex < questions.length - 1) {
    currentIndex += 1;
    persistCurrentQuestionIndex();
    renderCurrentQuestion(true);
  } else {
    markCurrentChapterComplete();
    setSummaryHtml('<p class="status-pass">You finished this chapter quiz. Use the next chapter link in the sidebar to continue.</p>');
    updateNavigationState();
    showInlineNext();
  }
}

function jumpToQuestion(index) {
  if (index < 0 || index >= questions.length) {
    return;
  }
  currentIndex = index;
  persistCurrentQuestionIndex();
  renderCurrentQuestion(true);
}

function persistCurrentQuestionIndex() {
  const chapter = window.__CHAPTER_DATA__;
  if (!chapter) {
    return;
  }
  updateChapterProgress(chapter.id, (progress) => ({
    ...progress,
    current_question_index: currentIndex,
  }), questions.length);
}

function persistCurrentAnswer() {
  const chapter = window.__CHAPTER_DATA__;
  const question = currentQuestion();
  if (!chapter || !question || !editor) {
    return;
  }
  updateChapterProgress(chapter.id, (progress) => ({
    ...progress,
    answers: {
      ...progress.answers,
      [question.id]: editor.getValue(),
    },
    current_question_index: currentIndex,
  }), questions.length);
}

function markQuestionCompleted(questionId) {
  const chapter = window.__CHAPTER_DATA__;
  if (!chapter || !questionId) {
    return;
  }
  updateChapterProgress(chapter.id, (progress) => {
    const completedQuestions = Array.from(new Set([...(progress.completed_questions || []), questionId]));
    return {
      ...progress,
      completed_questions: completedQuestions,
      completed: completedQuestions.length >= questions.length,
      current_question_index: currentIndex,
    };
  }, questions.length);
}

function markCurrentChapterComplete() {
  const chapter = window.__CHAPTER_DATA__;
  if (!chapter) {
    return;
  }
  updateChapterProgress(chapter.id, (progress) => ({
    ...progress,
    completed: true,
    completed_questions: questions.map((question) => question.id),
    current_question_index: currentIndex,
  }), questions.length);
}

function resetCurrentAnswer() {
  const chapter = window.__CHAPTER_DATA__;
  const question = currentQuestion();
  if (!chapter || !question) {
    return;
  }
  updateChapterProgress(chapter.id, (progress) => {
    const answers = { ...progress.answers };
    delete answers[question.id];
    return {
      ...progress,
      answers,
      current_question_index: currentIndex,
    };
  }, questions.length);
}

function openPdfDrawer(url) {
  const drawer = document.getElementById("pdf-drawer");
  const frame = document.getElementById("pdf-drawer-frame");
  if (!drawer || !frame) {
    return;
  }

  frame.src = url;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
}

function closePdfDrawer() {
  const drawer = document.getElementById("pdf-drawer");
  const frame = document.getElementById("pdf-drawer-frame");
  if (!drawer || !frame) {
    return;
  }

  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  frame.src = "about:blank";
}

function openTutorialDrawer() {
  const drawer = document.getElementById("tutorial-drawer");
  if (!drawer) {
    return;
  }

  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  window.requestAnimationFrame(() => {
    if (tutorialViewer && tutorialViewer.refresh) tutorialViewer.refresh();
    if (playgroundEditor && playgroundEditor.refresh) playgroundEditor.refresh();
  });
  window.setTimeout(() => {
    if (tutorialViewer && tutorialViewer.refresh) tutorialViewer.refresh();
    if (playgroundEditor && playgroundEditor.refresh) playgroundEditor.refresh();
  }, 220);
}

function closeTutorialDrawer() {
  const drawer = document.getElementById("tutorial-drawer");
  if (!drawer) {
    return;
  }

  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}

function setLessonCollapsed(collapsed, persist = false) {
  const panel = document.querySelector(".lesson-panel");
  const content = document.getElementById("lesson-content");
  const button = document.getElementById("lesson-collapse-toggle");
  if (!panel || !content || !button) {
    return;
  }

  panel.classList.toggle("is-collapsed", collapsed);
  content.hidden = collapsed;
  button.textContent = collapsed ? "Show Lesson" : "Collapse Lesson";
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");

  if (persist && window.__CHAPTER_DATA__) {
    updateChapterProgress(window.__CHAPTER_DATA__.id, (progress) => ({
      ...progress,
      lesson_collapsed: collapsed,
    }), questions.length);
  }
}

function restoreLessonCollapseState() {
  if (!window.__CHAPTER_DATA__) {
    return;
  }
  const progress = getChapterProgress(window.__CHAPTER_DATA__.id, questions.length);
  setLessonCollapsed(Boolean(progress.lesson_collapsed));
}

function toggleLessonCollapse() {
  const panel = document.querySelector(".lesson-panel");
  setLessonCollapsed(!(panel && panel.classList.contains("is-collapsed")), true);
}

function currentQuestion() {
  return questions[currentIndex];
}

function setSummaryHtml(html) {
  const element = document.getElementById("summary-output");
  if (element) {
    element.innerHTML = html;
  }
}

function setPlainOutput(id, text) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = text || "";
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatSnapshot(value) {
  if (value === null || value === undefined) {
    return "Not available";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

function renderResultDetails(result) {
  const inputSnapshot = escapeHtml(formatSnapshot(result.input_snapshot));
  const expectedOutput = escapeHtml(formatSnapshot(result.expected_output));
  const actualOutput = escapeHtml(formatSnapshot(result.actual_output));

  return [
    '<div class="test-details">',
    '<div class="test-detail-block"><strong>Input</strong><pre class="test-detail-pre">', inputSnapshot, "</pre></div>",
    '<div class="test-detail-block"><strong>Expected Output</strong><pre class="test-detail-pre">', expectedOutput, "</pre></div>",
    '<div class="test-detail-block"><strong>Your Output</strong><pre class="test-detail-pre">', actualOutput, "</pre></div>",
    "</div>",
  ].join("");
}

function bindHintButtons() {
  return;
}

function applyHint(key) {
  if (!key) {
    return;
  }
  const detail = document.querySelector(`[data-result-detail="${key}"]`);
  const counter = document.querySelector(`[data-hint-counter="${key}"]`);
  hintCounts[key] = (hintCounts[key] || 0) + 1;

  if (counter) {
    const remaining = Math.max(0, 3 - hintCounts[key]);
    counter.textContent = remaining > 0
      ? `Hint used ${hintCounts[key]} time(s). Reveal after ${remaining} more.`
      : "Input and expected output revealed below.";
  }

  if (detail && hintCounts[key] >= 3) {
    detail.hidden = false;
  }
}

function applyGlobalHint() {
  const detailShells = Array.from(document.querySelectorAll("[data-result-detail]"));
  if (detailShells.length === 0) {
    setSummaryHtml('<p class="status-fail">Run or submit the current question first, then use Hint.</p>');
    return;
  }

  detailShells.forEach((detail) => applyHint(detail.dataset.resultDetail));
  recordHintUsage();
  syncAnswerTab();
}

function toggleEditorComment() {
  if (!editor) {
    return;
  }
  if (typeof editor.execCommand === "function") {
    editor.execCommand("toggleComment");
  }
}

function editorOptions(baseOptions = {}) {
  return {
    mode: "python",
    theme: "material",
    lineNumbers: true,
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    ...baseOptions,
  };
}

function insertSoftTab(cm) {
  if (!cm || typeof cm.replaceSelection !== "function") {
    return;
  }
  cm.replaceSelection(" ".repeat(4), "end", "+input");
}

function outdentSelection(cm) {
  if (!cm || typeof cm.listSelections !== "function" || typeof cm.getLine !== "function") {
    return;
  }

  const targetLines = new Set();
  cm.listSelections().forEach((selection) => {
    const from = selection.from();
    const to = selection.to();
    const endLine = to.ch === 0 && to.line > from.line ? to.line - 1 : to.line;
    for (let line = from.line; line <= endLine; line += 1) {
      targetLines.add(line);
    }
  });

  cm.operation(() => {
    Array.from(targetLines).sort((a, b) => a - b).forEach((line) => {
      const text = cm.getLine(line);
      const match = text.match(/^(\t| {1,4})/);
      if (match) {
        cm.replaceRange("", { line, ch: 0 }, { line, ch: match[0].length }, "+input");
      }
    });
  });
}

function wrapSelections(cm, openChar, closeChar) {
  if (!cm || typeof cm.listSelections !== "function") {
    return false;
  }

  const selections = cm.listSelections();
  const hasSelectedText = selections.some((selection) => !selection.empty());
  if (!hasSelectedText) {
    return false;
  }

  const replacements = cm.getSelections().map((selection) => `${openChar}${selection}${closeChar}`);
  cm.replaceSelections(replacements, "around", "+input");
  return true;
}

function insertBracketPair(cm, openChar, closeChar) {
  if (!cm || typeof cm.getCursor !== "function" || typeof cm.replaceSelection !== "function") {
    return false;
  }

  const cursor = cm.getCursor();
  cm.replaceSelection(`${openChar}${closeChar}`, "around", "+input");
  cm.setCursor(cursor.line, cursor.ch + 1);
  return true;
}

function skipExistingCloseBracket(cm, closeChar) {
  if (!cm || typeof cm.getCursor !== "function" || typeof cm.getRange !== "function") {
    return false;
  }

  const cursor = cm.getCursor();
  const next = { line: cursor.line, ch: cursor.ch + 1 };
  if (cm.getRange(cursor, next) !== closeChar) {
    return false;
  }

  cm.setCursor(next);
  return true;
}

function getLineIndent(text) {
  const match = text.match(/^\s*/);
  return match ? match[0] : "";
}

function immediateBracketPairContext(cm) {
  if (!cm || typeof cm.getCursor !== "function" || typeof cm.getLine !== "function") {
    return null;
  }
  if (typeof cm.somethingSelected === "function" && cm.somethingSelected()) {
    return null;
  }

  const pairs = {
    "(": ")",
    "[": "]",
    "{": "}",
  };
  const cursor = cm.getCursor();
  const line = cm.getLine(cursor.line);
  const before = line[cursor.ch - 1];
  const after = line[cursor.ch];
  if (!Object.prototype.hasOwnProperty.call(pairs, before) || pairs[before] !== after) {
    return null;
  }

  return { cursor, line, before, after };
}

function splitImmediateBracketPair(cm, context = immediateBracketPairContext(cm)) {
  if (!context) {
    return false;
  }

  const baseIndent = getLineIndent(context.line);
  const innerIndent = `${baseIndent}${" ".repeat(4)}`;
  cm.replaceSelection(`\n${innerIndent}\n${baseIndent}`, "around", "+input");
  cm.setCursor(context.cursor.line + 1, innerIndent.length);
  return true;
}

function bindEnterKey(cm) {
  if (!cm || typeof cm.getWrapperElement !== "function") {
    return;
  }

  cm.getWrapperElement().addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }

    const pairContext = immediateBracketPairContext(cm);
    if (!pairContext) {
      return;
    }

    event.preventDefault();
    if (typeof event.stopImmediatePropagation === "function") {
      event.stopImmediatePropagation();
    } else {
      event.stopPropagation();
    }
    splitImmediateBracketPair(cm, pairContext);
  }, true);
}

function bindBracketKeys(cm) {
  if (!cm || typeof cm.on !== "function") {
    return;
  }

  const pairs = {
    "(": ")",
    "[": "]",
    "{": "}",
  };
  const closingChars = new Set(Object.values(pairs));

  cm.on("keydown", (_, event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }

    if (closingChars.has(event.key) && skipExistingCloseBracket(cm, event.key)) {
      event.preventDefault();
      return;
    }

    const closeChar = pairs[event.key];
    if (!closeChar) {
      return;
    }
    if (wrapSelections(cm, event.key, closeChar)) {
      event.preventDefault();
      return;
    }
    if (insertBracketPair(cm, event.key, closeChar)) {
      event.preventDefault();
    }
  });
}

function installEditorKeys(cm, extraKeys = {}) {
  if (!cm || typeof cm.addKeyMap !== "function") {
    return;
  }

  cm.addKeyMap({
    Tab: () => { insertSoftTab(cm); },
    "Shift-Tab": () => { outdentSelection(cm); },
    ...extraKeys,
  });
  bindEnterKey(cm);
  bindBracketKeys(cm);
}

function normalizeEditorIndentation(cm) {
  if (!cm || typeof cm.getValue !== "function" || typeof cm.setValue !== "function") {
    return;
  }

  const rawCode = cm.getValue();
  const normalizedCode = rawCode.replace(/^\t+/gm, (tabs) => " ".repeat(tabs.length * 4));
  if (normalizedCode !== rawCode) {
    const cursor = typeof cm.getCursor === "function" ? cm.getCursor() : null;
    cm.setValue(normalizedCode);
    if (cursor && typeof cm.setCursor === "function") {
      cm.setCursor(cursor);
    }
  }
}

function buildEditor() {
  const textarea = document.getElementById("code-editor");
  const tutorialTextarea = document.getElementById("tutorial-script-editor");
  const playgroundTextarea = document.getElementById("playground-editor");
  if (!textarea) {
    return;
  }

  if (window.CodeMirror) {
    if (tutorialTextarea) {
      tutorialViewer = window.CodeMirror.fromTextArea(tutorialTextarea, editorOptions({
        readOnly: "nocursor",
        lineWrapping: false,
      }));
    }
    if (playgroundTextarea) {
      playgroundEditor = window.CodeMirror.fromTextArea(playgroundTextarea, editorOptions());
      installEditorKeys(playgroundEditor);
    }
    editor = window.CodeMirror.fromTextArea(textarea, editorOptions());
    installEditorKeys(editor, {
      "Cmd-Enter": () => { triggerPrimaryShortcut(); },
      "Ctrl-Enter": () => { triggerPrimaryShortcut(); },
      "Shift-Enter": () => { triggerSubmit(); },
      "Cmd-/": () => { toggleEditorComment(); },
      "Ctrl-/": () => { toggleEditorComment(); },
    });
    editor.on("change", () => {
      const question = currentQuestion();
      lastRunPassed = false;
      lastRunQuestionId = question ? question.id : null;
      persistCurrentAnswer();
      syncInlineNextVisibility();
    });
  } else {
    tutorialViewer = tutorialTextarea ? { refresh: () => {} } : null;
    playgroundEditor = playgroundTextarea ? {
      getValue: () => playgroundTextarea.value,
      setValue: (value) => { playgroundTextarea.value = value; },
      refresh: () => {},
    } : null;
    editor = {
      getValue: () => textarea.value,
      setValue: (value) => { textarea.value = value; },
      refresh: () => {},
    };
  }
}

function renderCurrentQuestion(resetCode = false) {
  const question = currentQuestion();
  if (!question) {
    setSummaryHtml("<p>No questions configured for this chapter yet.</p>");
    updateNavigationState();
    return;
  }

  document.getElementById("question-title").textContent = question.title;
  document.getElementById("question-prompt").textContent = question.prompt;
  document.getElementById("visible-examples-title").textContent = question.visible_examples_title || chapterUi().question_prompt_title || "Visible Checks";
  document.getElementById("answer-editor-title").textContent = question.answer_editor_title || chapterUi().answer_editor_title || "Your Answer";
  document.getElementById("answer-editor-note").textContent = question.answer_editor_note || chapterUi().answer_editor_note || "Write the code for this question here";

  const examples = document.getElementById("visible-examples");
  examples.innerHTML = "";
  question.visible_examples.forEach((example) => {
    const item = document.createElement("li");
    item.textContent = example;
    examples.appendChild(item);
  });

  if (resetCode && editor) {
    const chapter = window.__CHAPTER_DATA__;
    const progress = chapter ? getChapterProgress(chapter.id, questions.length) : null;
    const savedCode = progress && progress.answers ? progress.answers[question.id] : undefined;
    editor.setValue(savedCode ?? question.starter_code);
    if (editor.refresh) editor.refresh();
  }

  if (tutorialViewer && tutorialViewer.refresh) tutorialViewer.refresh();
  if (playgroundEditor && playgroundEditor.refresh) playgroundEditor.refresh();

  setSummaryHtml(`Question ${currentIndex + 1} of ${questions.length}<br><strong>${question.title}</strong>`);
  setPlainOutput("stdout-output", "");
  setPlainOutput("stderr-output", "");
  hintCounts = {};
  lastRunPassed = false;
  lastRunQuestionId = question.id;
  syncInlineNextVisibility();
  syncAnswerTab();
  updateNavigationState();
  persistCurrentQuestionIndex();
}

function formatResults(payload) {
  if (payload.error) {
    return `<p class="status-fail">${payload.error}</p>`;
  }

  hintCounts = {};
  const statusClass = payload.ok ? "status-pass" : "status-fail";
  const label = payload.mode === "submit" ? "Hidden tests" : "Visible tests";
  const items = payload.results
    .map((result, index) => {
      const resultKey = `${payload.mode}-${currentIndex}-${index}`;
      const status = result.passed ? "PASS" : "FAIL";
      const message = result.message ? ` - ${result.message}` : "";
      const showDetailsImmediately = payload.mode === "submit" && result.passed;
      const safeHint = escapeHtml(result.hint || "Compare your output to the expected behavior.");
      return `
        <li class="test-result-item">
          <div><strong>${status}</strong> ${escapeHtml(result.name)}${escapeHtml(message)}</div>
          <div class="test-hint-row">
            <span class="hint-text">${safeHint}</span>
          </div>
          <div class="hint-counter" data-hint-counter="${resultKey}">Press hint 3 times to reveal the test input and expected output.</div>
          <div class="test-detail-shell" data-result-detail="${resultKey}" ${showDetailsImmediately ? "" : "hidden"}>
            ${renderResultDetails(result)}
          </div>
        </li>
      `;
    })
    .join("");

  return `<p class="${statusClass}">${label}: ${payload.passed}/${payload.total} passed</p><ul>${items}</ul>`;
}

function isSuccessfulSubmit(payload, mode) {
  if (mode !== "submit" || !payload || payload.error) {
    return false;
  }
  if (payload.ok === true) {
    return true;
  }
  if (typeof payload.passed === "number" && typeof payload.total === "number" && payload.total > 0) {
    return payload.passed === payload.total;
  }
  return false;
}

async function sendCode(mode) {
  const chapter = window.__CHAPTER_DATA__;
  const question = currentQuestion();
  normalizeEditorIndentation(editor);
  const response = await fetch(`/api/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chapter_id: chapter.id,
      question_id: question.id,
      code: editor.getValue(),
    }),
  });

  const payload = await response.json();
  if (payload.autofixed && typeof payload.normalized_code === "string" && editor) {
    editor.setValue(payload.normalized_code);
  }
  setSummaryHtml(formatResults(payload));
  bindHintButtons();
  setPlainOutput("stdout-output", payload.stdout);
  setPlainOutput("stderr-output", payload.stderr || payload.error || "");
  if (mode === "run") {
    lastRunPassed = Boolean(payload.ok);
    lastRunQuestionId = question.id;
  } else if (mode === "submit") {
    lastRunQuestionId = question.id;
  }
  if (isSuccessfulSubmit(payload, mode)) {
    if (mode === "submit") {
      markQuestionCompleted(question.id);
      if (currentIndex === questions.length - 1) {
        markCurrentChapterComplete();
      }
    }
    showInlineNext();
  } else {
    syncInlineNextVisibility();
  }
}

async function runPlayground() {
  if (!playgroundEditor) {
    return;
  }
  const response = await fetch("/api/playground", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: playgroundEditor.getValue() }),
  });

  const payload = await response.json();
  setPlainOutput("playground-stdout", payload.stdout);
  setPlainOutput("playground-stderr", payload.stderr || payload.error || "");
}

function bindQuizUi() {
  const actionHandlers = {
    "run-code": triggerRun,
    "submit-code": triggerSubmit,
    "show-hint": applyGlobalHint,
    "reset-question": () => {
      resetCurrentAnswer();
      renderCurrentQuestion(true);
    },
    "previous-question": () => {
      if (currentIndex > 0) {
        currentIndex -= 1;
        renderCurrentQuestion(true);
      }
    },
    "next-question": goToNextQuestion,
    "open-tutorial-drawer": openTutorialDrawer,
    "open-pdf-drawer": (event) => openPdfDrawer(event.currentTarget.dataset.pdfUrl),
    "toggle-lesson-collapse": toggleLessonCollapse,
    "run-playground": runPlayground,
    "show-answer": revealAnswer,
    "reset-playground": () => {
      if (!playgroundEditor) {
        return;
      }
      playgroundEditor.setValue(window.__CHAPTER_DATA__.playground_starter || "");
      setPlainOutput("playground-stdout", "");
      setPlainOutput("playground-stderr", "");
    },
  };

  Object.entries(actionHandlers).forEach(([action, handler]) => {
    findActionElements(action).forEach((element) => {
      element.addEventListener("click", handler);
    });
  });

  document.querySelectorAll("[data-question-jump]").forEach((element) => {
    element.addEventListener("click", () => {
      jumpToQuestion(Number(element.dataset.questionJump));
    });
  });

  const closeButton = document.getElementById("tutorial-drawer-close");
  if (closeButton) closeButton.addEventListener("click", closeTutorialDrawer);

  const pdfCloseButton = document.getElementById("pdf-drawer-close");
  if (pdfCloseButton) pdfCloseButton.addEventListener("click", closePdfDrawer);

  window.addEventListener("keydown", (event) => {
    const activeElement = document.activeElement;
    const codeMirrorActive = activeElement && activeElement.closest && activeElement.closest(".CodeMirror");
    if (!codeMirrorActive) {
      return;
    }

    if (event.key === "Enter") {
      if (event.shiftKey) {
        event.preventDefault();
        triggerSubmit();
        return;
      }
      if (event.metaKey || event.ctrlKey) {
        event.preventDefault();
        triggerPrimaryShortcut();
      }
      return;
    }

  });
}

function initializeChapterApp() {
  refreshSavedProgressUi();
  if (!window.__CHAPTER_DATA__) {
    return;
  }

  questions = window.__CHAPTER_DATA__.questions || [];
  const savedProgress = getChapterProgress(window.__CHAPTER_DATA__.id, questions.length);
  if (questions.length > 0) {
    currentIndex = Math.min(Math.max(savedProgress.current_question_index || 0, 0), questions.length - 1);
  }
  restoreLessonCollapseState();
  buildEditor();
  bindQuizUi();
  renderCurrentQuestion(true);
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", initializeChapterApp);
} else {
  initializeChapterApp();
}
