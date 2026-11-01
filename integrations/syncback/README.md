# Подключение конвейера к SyncBack

Наш конвейер встаёт в SyncBack как ещё одна реализация их порта `LLMClient`
(`app/domain/interfaces/llm_client.py`). API, воркер, маппер правок, экспорт и UI не меняются:
включается переменной `LLM_PROVIDER=ml_pipeline`.

Сверено с версией SyncBack от 29.10.2026 (архив `SyncBack-main 1.zip`). В их репозиторий мы ничего не пишем:
ниже то, что нужно сделать на их стороне.

## Что делает

`generate_suggestions(document_text, source_text, document_format)`:

1. документ режем на абзацы. Для docx это строки их `plain_text`, заголовок раздела угадываем по короткой строке без точки;
2. источник (release notes, протокол, страница) режем на куски, а список — на отдельные пункты;
3. для каждого абзаца находим 3 ближайших куска источника (BGE-M3 + BM25);
4. сначала короткий вызов модели «только вердикт», полный вызов с правкой — только если абзац противоречит источнику;
5. ответ отдаём их же `LLMSuggestionItem`: `change_type="modify"`, `old_text` — абзац дословно, в `rationale` цитата из источника.

То, что их маппер всё равно отбросит (`old_text` не дословно или с переводом строки в docx), мы не отдаём.

## Шаги на стороне SyncBack

1. Поставить наш пакет в окружение воркера:
   ```
   pip install "syncscribe-ml @ git+https://github.com/rishotkaB/syncscribe-ml"
   ```
   Репозиторий приватный: нужен доступ на чтение или собранный wheel (`pip wheel .` в нашем репо).
2. Скопировать `ml_pipeline_client.py` в `app/infrastructure/llm/`.
3. `app/core/config.py`: добавить `"ml_pipeline"` в `Literal` поля `llm_provider`.
4. `app/infrastructure/llm/factory.py`:
   ```python
   if settings.llm_provider == "ml_pipeline":
       from app.infrastructure.llm.ml_pipeline_client import MLPipelineClient
       return MLPipelineClient(settings)
   ```
5. Переменные окружения:
   ```
   LLM_PROVIDER=ml_pipeline
   LLM_ENDPOINT=http://host.docker.internal:11434/v1   # тот же OpenAI-совместимый API: Ollama или vLLM
   LLM_MODEL=qwen3:8b
   LLM_AUTH=none                                        # или bearer + LLM_API_KEY
   EMBED_MODEL=bge-m3                                   # для Ollama: ollama pull bge-m3
   EMBED_BASE_URL=                                      # если эмбеддинги на другом сервере; пусто = LLM_ENDPOINT
   ML_PIPELINE_WORKERS=1                                # для vLLM 8–16: абзацы проверяются параллельно
   ```

## Проверка без SyncBack-сервисов

`check_with_syncback.py` берёт код SyncBack из папки (только читает) и прогоняет цепочку:
настоящий .docx → их `DocxParser` → наш адаптер → их `map_to_suggestions`.
Печатает, сколько правок их маппер принял.

```
pip install sqlalchemy pydantic-settings python-docx
python integrations/syncback/check_with_syncback.py --syncback <папка SyncBack-main> --document data/large/truth/derived.md --source data/syncback/release_notes.md --endpoint http://localhost:11434/v1 --model qwen3:8b
```

## Что пока не покрыто (вопросы к продукту и разработке)

- **Правки `add` и `delete`.** Мы отвечаем только на вопрос «стал ли абзац неверным». Вставку нового абзаца
  о новой функции («добавлена поддержка PDF») не предлагаем.
- **Пометки без правки** (источник противоречит сам себе) в формате SyncBack выразить нечем. Сейчас они
  уходят только в лог и в `raw_response`, который SyncBack не сохраняет.
- **Споры между источниками.** Воркер SyncBack вызывает модель отдельно на каждый источник, поэтому
  противоречие между двумя источниками внутри `generate_suggestions` не видно. Чтобы его ловить, источники
  анализа нужно передавать вместе.
- **Время.** Лимит задачи у них 540 с. Документ на 200 абзацев на RTX 4060 обрабатывается ~4–5 минут.
  Больше 400 абзацев на такой карте может не уложиться; на сервере с vLLM и `ML_PIPELINE_WORKERS=8` — с запасом.
- **`LLM_AUTH=oauth_gigachat`** не поддержан (и GigaChat внешний, для on-prem не нужен).
