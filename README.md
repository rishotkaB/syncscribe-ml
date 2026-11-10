# syncscribe-ml

ML-часть SyncScribe: находит в документе абзацы, которые противоречат источнику истины (протоколу встречи, release notes, руководству), и предлагает правки.

## Быстрый старт: контракт 2.0 с SyncBack

Что делает: JSON с документом, разбитым на блоки, и источником истины → JSON с правками `modify` / `add`, `confidence_score` и пометками «проверьте вручную». Поля и значения: [`docs/contract_v2_schema.md`](docs/contract_v2_schema.md), как устроено внутри: [`docs/contract_v2.md`](docs/contract_v2.md).

### Что отправлять и что придёт

Обмен файлами: **один JSON на вход на один документ → один JSON на выход**. Полный справочник полей, типов и ошибок: [`docs/contract_v2_schema.md`](docs/contract_v2_schema.md).

**Вход.** Документ, уже разбитый на блоки, и текст источника истины:

```json
{
  "request_id": "11111111-1111-4111-8111-111111111111",
  "analysis_job_id": "22222222-2222-4222-8222-222222222222",
  "document": {
    "id": "33333333-3333-4333-8333-333333333333",
    "name": "Регламент поддержки.docx",
    "format": "docx",
    "blocks": [
      {"block_id": "d-1", "type": "heading",   "text": "Сроки", "editable": false},
      {"block_id": "d-2", "type": "paragraph", "text": "Срок ответа на обращение — 10 рабочих дней.", "editable": true},
      {"block_id": "d-3", "type": "heading",   "text": "Оплата", "editable": false},
      {"block_id": "d-4", "type": "paragraph", "text": "Оплата принимается картой.", "editable": true}
    ]
  },
  "source": {
    "id": "44444444-4444-4444-8444-444444444444",
    "name": "Решения встречи от 02.11.2026",
    "type": "file",
    "text": "Решения:\n- Срок ответа на обращение: 5 рабочих дней.\n- Добавлена оплата по счёту для юридических лиц."
  }
}
```

Что важно во входе:
- **Блоки** идут в порядке документа, `block_id` уникален. Правки в ответе ссылаются на него.
- **`text` блока** — ровно как в документе: `old_text` и `insert_after` в ответе вырезаются из него дословно.
- **`type: "heading"`** не проверяется, а задаёт раздел для блоков ниже. **`editable: false`** — блок не трогаем.
- **Источник** — просто текст. Списки (`-`, `•`, `1.`) делим на пункты, пустые строки — на абзацы. `name` показывается пользователю в обосновании правки.
- **Несколько источников** — одним файлом, списком `"sources": [{...}, {...}]` вместо `"source"`. Только так видно, что источники противоречат друг другу. Если слать каждый источник отдельным запросом, споры между ними не найдутся.
- `request_id` и `analysis_job_id` возвращаются в ответе без изменений, по ним связывать запрос и ответ.

**Выход.** Для примера выше:

```json
{
  "contract_version": "2.0",
  "request_id": "11111111-1111-4111-8111-111111111111",
  "analysis_job_id": "22222222-2222-4222-8222-222222222222",
  "status": "completed",
  "suggestions": [
    {
      "suggestion_id": "s-1",
      "change_type": "modify",
      "target_block_id": "d-2",
      "old_text": "Срок ответа на обращение — 10 рабочих дней.",
      "new_text": "Срок ответа на обращение — 5 рабочих дней.",
      "insert_after": null,
      "rationale": "По решениям встречи срок ответа на обращение составляет 5 рабочих дней, а не 10. Источник: «Срок ответа на обращение: 5 рабочих дней.» (Решения встречи от 02.11.2026)",
      "confidence_score": 0.95
    },
    {
      "suggestion_id": "s-2",
      "change_type": "add",
      "target_block_id": "d-4",
      "old_text": null,
      "new_text": "Юридические лица также могут оплатить по счёту.",
      "insert_after": "Оплата принимается картой.",
      "rationale": "В решениях встречи добавлена оплата по счёту для юридических лиц. Источник: «Добавлена оплата по счёту для юридических лиц.» (Решения встречи от 02.11.2026)",
      "confidence_score": 0.8
    }
  ],
  "flags": [],
  "meta": {"blocks": 4, "checked_blocks": 2, "seconds_total": 13.0},
  "error": null
}
```

Как читать выход:
- **`status`**: `completed` — правки можно показывать; `failed` — смотреть `error` (`invalid_request` — запрос не по формату, `pipeline_error` — сбой модели или поиска), `suggestions` тогда пустой.
- **`modify`**: найти `old_text` в блоке `target_block_id` и заменить на `new_text`. `old_text` — весь блок или одно предложение из него, всегда дословно.
- **`add`**: вставить `new_text` сразу после `insert_after` (последнее предложение блока `target_block_id`). `old_text` = `null`.
- **`delete`** зарезервирован, сейчас не приходит.
- **`confidence_score`** 0–1: 0.95–0.97 — заменено одно значение и оно есть в цитате источника; до 0.8 — добавление; до 0.75 — переписана формулировка; 0.6 и ниже — есть сомнение, причина дописана в конце `rationale` после «Проверьте внимательно:». Порог показа в интерфейсе — решение продукта.
- **`flags`** — пометки «проверьте вручную» без готовой правки: источники противоречат друг другу по этому блоку, в `quotes` цитата из каждого. Поле сверх договорённого, его можно не показывать.
- **`meta`** — статистика прогона, только для логов; состав ключей может меняться.
- Правки отсортированы по порядку блоков. Пустой `suggestions` при `completed` — документ соответствует источнику.

**Как вызывать.** Файлами:
```
python scripts/run_contract.py запрос.json -o ответ.json
```
Или из Python, если ML-часть стоит в том же окружении (на вход и выход — обычные `dict`):
```python
from syncscribe_ml.contract import run
response = run(request, model="qwen3:8b")
```
`run` не бросает исключений: любая ошибка возвращается как `status: "failed"`.

Нужны две модели:
- **LLM** — локальная qwen3:8b через Ollama или любая модель по OpenAI-совместимому API;
- **BGE-M3** (эмбеддинги для поиска) — через Ollama или прямо в Python. Выбирается флагом `--embed`.

### 1. Окружение (Python 3.10+, команды из папки репозитория)

```
python -m venv .venv
.venv\Scripts\activate          # Windows; на Linux/macOS: source .venv/bin/activate
```

Дальше один из двух вариантов.

**Вариант А: с Ollama.** Подходит, если Ollama уже стоит (на видеокарте быстрее всего).
```
pip install -e .
ollama pull bge-m3
ollama pull qwen3:8b             # только если LLM тоже локальная
```

**Вариант Б: без Ollama.** BGE-M3 считается в Python (на видеокарте, если она есть, иначе на процессоре), LLM — по API.
```
pip install -e .[local]
```
Ставит PyTorch и sentence-transformers (~2–3 ГБ). При первом запуске модель BGE-M3 скачается с Hugging Face (~2,3 ГБ) и дальше берётся из кэша.

### 2. Запуск на примере

`-o` — куда записать ответ. Свой запрос: тот же скрипт, путь к своему JSON.

Локальная LLM (вариант А):
```
python scripts/run_contract.py data/contract/example_request.json -o out.json
```

LLM по API (PowerShell; ключ только через переменную, не флагом):
```
$env:LLM_API_KEY = "sk-..."
python scripts/run_contract.py data/contract/example_request.json -o out.json --provider openai --base-url https://api.openai.com/v1 --model gpt-4o-mini --workers 8
```
На Linux/macOS вместо первой строки: `export LLM_API_KEY=sk-...`

`--embed` выбирает, чем считать эмбеддинги: `auto` (по умолчанию: в Python, если установлен вариант Б, иначе Ollama), `ollama` или `local`. Что выбрано, скрипт пишет в первой строке вывода.

Большой пример с разметкой (209 блоков, 2 источника, 42 ошибки): `data/contract/large_request.json` с `--labels data/contract/large_labels.json`.

**Облачные API — только для разработки и только на синтетических данных.** Документы клиентов под NDA отправляются только в on-prem модель.

### Если не работает

| Ошибка | Что сделать |
|---|---|
| `Failed to connect to Ollama` | Запусти Ollama или поставь вариант Б и добавь `--embed local` |
| `401 Unauthorized` | Неверный ключ: нужен API-ключ (для OpenAI — platform.openai.com) и баланс; переменная задана в том же окне терминала |
| `model "bge-m3" not found` | `ollama pull bge-m3` |
| Долго на первом запуске с `--embed local` | Скачивается BGE-M3 (~2,3 ГБ), это один раз |

## HTTP-сервис для SyncBack

SyncBack вызывает ML-часть по HTTP: тот же `run()`, обёрнутый в сервис (`src/syncscribe_ml/server.py`). Логика анализа не меняется.

```
pip install -e .[server]
uvicorn syncscribe_ml.server:app --host 0.0.0.0 --port 8000
```
Или в Docker (эмбеддинги через Ollama; для BGE-M3 в Python — `--build-arg EXTRAS=server,local`):
```
docker build -t syncscribe-ml .
docker run -p 8000:8000 -e OLLAMA_HOST=http://ollama:11434 -e SYNCSCRIBE_ML_MODEL=qwen3:8b syncscribe-ml
```

- `POST /v1/analyze` — тело: запрос 2.0; ответ: `200` и ответ 2.0. Ошибки запроса и модели — в теле (`status: "failed"`, `invalid_request` / `pipeline_error`). HTTP-ошибки только интеграционные: `400` — не JSON-объект, `401` — неверный ключ, `429` + `Retry-After` — все слоты заняты (SyncBack повторит).
- `GET /health` — `200`, когда сервис принимает запросы. Без авторизации.
- Прогон держит соединение до ответа (большой документ — минуты). SyncBack ждёт до 25 минут.

Параметры анализа задаются переменными окружения сервиса, SyncBack их не передаёт:

| Переменная | По умолчанию | Что |
|---|---|---|
| `SYNCSCRIBE_ML_MODEL` | `LLM_MODEL` или `qwen3:8b` | модель LLM |
| `SYNCSCRIBE_ML_WORKERS` | 1 | блоков параллельно внутри запроса |
| `SYNCSCRIBE_ML_TOP_K` | 3 | кусков источника на блок |
| `SYNCSCRIBE_ML_MODE` | `hybrid` | `hybrid` / `dense` / `bm25` |
| `SYNCSCRIBE_ML_FAST` | 1 | сначала короткий вызов «только вердикт» |
| `SYNCSCRIBE_ML_ADDS` | 1 | искать правки «добавить» |
| `SYNCSCRIBE_ML_MAX_CONCURRENT` | 1 | запросов одновременно, сверх — `429` |
| `SYNCSCRIBE_ML_RETRY_AFTER` | 30 | секунд в `Retry-After` |
| `SYNCSCRIBE_ML_API_KEY` | пусто | если задан — нужен `Authorization: Bearer <ключ>` |
| `SYNCSCRIBE_ML_WARMUP` | 0 | загрузить модель эмбеддингов на старте |
| `SYNCSCRIBE_ML_PROMPTS_DIR` | `prompts/` репозитория | папка промптов (в образе — `/app/prompts`) |

Провайдер LLM и эмбеддинги — как раньше: `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `EMBED_BACKEND`, `OLLAMA_HOST`.

Проверка без модели: `python tests/test_server.py`.

## Другие режимы и eval

### Запуск eval (Windows, локально)

1. Установи [Ollama](https://ollama.com/download) и скачай модель:
   ```
   ollama pull qwen3:8b
   ollama pull bge-m3
   ```
2. Подготовь окружение (Python 3.10+). Команды выполняются из папки репозитория:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```
3. Сначала прогони 3 примера, чтобы убедиться, что всё работает, потом весь набор:
   ```
   python scripts/run_eval.py --model qwen3:8b --limit 3
   python scripts/run_eval.py --model qwen3:8b
   ```

## Модель: локально или по API

Все скрипты, которые зовут модель, принимают `--provider` и `--base-url`:

- `--provider ollama` (по умолчанию) — локальная Ollama;
- `--provider openai --base-url <адрес>/v1` — любой OpenAI-совместимый API: vLLM на сервере, Ollama через `/v1`, облачные API. Ключ задаётся переменной `LLM_API_KEY`, не флагом.

Пример (PowerShell):
```
$env:LLM_API_KEY = "ключ"
python scripts/eval_truth.py --provider openai --base-url https://openrouter.ai/api/v1 --model qwen/qwen3-14b
```
Те же настройки можно задать переменными `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`.

**Облачные API — только для разработки и только на синтетических данных.** Данные клиентов под NDA отправляются только в on-prem модель.

## Что где

| Путь | Что это |
|---|---|
| `data/eval/gold_v0.jsonl` | Eval-набор: 20 синтетических примеров, описание в `data/eval/README.md` |
| `prompts/impact_v1.md` | Системный промпт. Новая версия промпта кладётся новым файлом (`impact_v2.md`), старый не трогаем |
| `src/syncscribe_ml/schema.py` | Схема ответа модели (JSON Schema для Ollama/vLLM и валидация через pydantic) |
| `src/syncscribe_ml/llm.py` | Один вызов модели на пару (изменение источника, фрагмент) |
| `src/syncscribe_ml/metrics.py` | Accuracy, precision, recall, traceability, латентность |
| `scripts/run_eval.py` | Прогон модели на наборе, результаты сохраняются в `results/` |
| `src/syncscribe_ml/diff.py` | Этап 3: что изменилось между версиями источника, и пары (БЫЛО, СТАЛО) по разделам для LLM |
| `src/syncscribe_ml/parse.py` | Загрузка .md/.txt, а .docx/.pdf через Docling (`pip install -e .[docling]`) |
| `scripts/diff_docs.py` | Показать изменения между двумя версиями (`--pairs` — что уйдёт в LLM) |
| `src/syncscribe_ml/retrieval.py` | Этап 4: гибридный поиск кандидатов (BGE-M3 через Ollama или sentence-transformers + BM25, слияние RRF) и правило для удалённых разделов |
| `src/syncscribe_ml/contract.py` | Контракт 2.0 с SyncBack: разбор запроса, проверка блоков, добавления, confidence_score, ответ |
| `src/syncscribe_ml/server.py`, `Dockerfile` | HTTP-сервис контракта 2.0 для SyncBack: `POST /v1/analyze`, `GET /health` |
| `scripts/run_contract.py` | Запуск контракта 2.0 на JSON-запросе (`--labels` — точность и полнота) |
| `scripts/md_to_request.py` | Собрать JSON-запрос из markdown-документа и файлов-источников |
| `data/contract/` | Примеры запросов и разметка для контракта 2.0 |
| `scripts/check_docs.py` | Сквозной прогон: две версии источника + производный документ → что править (`--all` — без поиска, `--json out.json` — результат в формате SyncBack, `--fast` — сначала короткий «да/нет», `--labels` — точность и полнота по разметке) |
| `scripts/make_large_example.py` | Большой синтетический пример для замера скорости: `data/large/` (источник 15 разделов, 13 изменений; документ 198 абзацев, 39 затронуты) |
| `scripts/check_truth.py` | Режим «источник истины»: документ + один или несколько файлов с верными данными → правки (`contradicts`) и пометки «источники спорят» без правки (`--fast`, `--labels`, `--json`) |
| `scripts/eval_truth.py` | Режим «источник истины» на eval-наборах: модель видит только новую версию источника, сравнение с режимом diff |
| `src/syncscribe_ml/syncback.py` | Перевод наших правок в формат ответа бэкенда SyncBack (`{"suggestions": [...]}`) |
| `scripts/eval_retrieval.py` | recall@k поиска на 50 примерах (`--mode hybrid/dense/bm25`) |
| `data/diff/` | Пример для этапа 3: руководство v1/v2 и производный документ |
| `tests/` | Проверки без модели: `python tests/test_contract.py`, `test_server.py`, `test_truth.py`, `test_diff.py`, `test_retrieval.py`, `test_syncback.py` |

## Пороги для пилота

Accuracy ≥ 80%, Precision ≥ 75%, Recall ≥ 75%, Hallucination ≤ 10%, Edit Usefulness ≥ 3.5, Traceability ≥ 60%.
