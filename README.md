# syncscribe-ml

ML-часть SyncScribe. Модуль определяет, затронут ли фрагмент документа изменением источника, и предлагает черновик правки.

## Запуск (Windows, локально)

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
| `src/syncscribe_ml/retrieval.py` | Этап 4: гибридный поиск кандидатов (BGE-M3 через Ollama + BM25, слияние RRF) и правило для удалённых разделов |
| `scripts/check_docs.py` | Сквозной прогон: две версии источника + производный документ → что править (`--all` — без поиска, `--json out.json` — результат в формате SyncBack, `--fast` — сначала короткий «да/нет», `--labels` — точность и полнота по разметке) |
| `scripts/make_large_example.py` | Большой синтетический пример для замера скорости: `data/large/` (источник 15 разделов, 13 изменений; документ 198 абзацев, 39 затронуты) |
| `scripts/check_truth.py` | Режим «источник истины»: документ + один или несколько файлов с верными данными → правки (`contradicts`) и пометки «источники спорят» без правки (`--fast`, `--labels`, `--json`) |
| `scripts/eval_truth.py` | Режим «источник истины» на eval-наборах: модель видит только новую версию источника, сравнение с режимом diff |
| `src/syncscribe_ml/syncback.py` | Перевод наших правок в формат ответа бэкенда SyncBack (`{"suggestions": [...]}`) |
| `scripts/eval_retrieval.py` | recall@k поиска на 50 примерах (`--mode hybrid/dense/bm25`) |
| `data/diff/` | Пример для этапа 3: руководство v1/v2 и производный документ |
| `tests/` | Проверки diff и поиска: `python tests/test_diff.py`, `python tests/test_retrieval.py`, `python tests/test_syncback.py` |

## Пороги для пилота

Accuracy ≥ 80%, Precision ≥ 75%, Recall ≥ 75%, Hallucination ≤ 10%, Edit Usefulness ≥ 3.5, Traceability ≥ 60%.
