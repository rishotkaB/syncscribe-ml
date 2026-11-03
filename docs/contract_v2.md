# Контракт 2.0 с SyncBack: обмен JSON-файлами

Ветка `contract-v2`. Договорённость 03.11.2026: от SyncBack приходит один JSON на документ, от нас уходит один JSON с ответом.

## Вход

Документ уже разбит на блоки (`document.blocks`), к ним привязываются правки. Источник — `source` (один объект) или `sources` (список, если источников несколько: тогда видны споры между ними). Пример: `data/contract/example_request.json`.

Проверяем только блоки с `editable: true` и `type` не `heading`. Заголовки задают раздел для следующих блоков.

## Выход

Поля по договорённости: `contract_version`, `request_id`, `analysis_job_id`, `status` (`completed` или `failed`), `suggestions`, `error`. Каждая правка содержит `suggestion_id`, `change_type`, `target_block_id`, `old_text`, `new_text`, `insert_after`, `rationale`, `confidence_score`.

Сверх договорённого (SyncBack может игнорировать):
- `flags` — пометки «проверьте вручную», если источники спорят между собой. Правку не предлагаем, даём цитаты из каждого источника (решение от 29.10, пока продукт не решил иначе).
- `meta` — статистика прогона: вызовы модели, время, вердикты.

`analysis_job_id` в ответе тот же, что во входе.

## Какие правки умеем

| change_type | Когда | old_text / insert_after |
|---|---|---|
| modify | блок противоречит источнику | текст блока или одно изменённое предложение, всегда дословно из блока |
| add | в источнике есть факт, которого в документе нет совсем | `insert_after` = последнее предложение блока `target_block_id` |
| delete | пока не предлагаем | нужен пример из реальных данных |

## confidence_score

1. Модель ставит оценку в полном вызове по шкале из промпта коллеги (0.90–1.00 / 0.70–0.89 / 0.40–0.69 / 0.10–0.39), промпты `prompts/truth_v3.md` и `prompts/add_v1.md`.
2. Поверх оценки модели — правила в коде:
   - цитата не нашлась в источнике дословно → не выше 0.4;
   - правка совпадает с текущим текстом → правку не отдаём.

Оценка не откалибрована: на qwen3:8b раньше уверенность была почти всегда 0.9–0.95. Калибровать будем по реальным принятым и отклонённым правкам или через судью.

## Запуск

```
python scripts/run_contract.py data/contract/example_request.json -o out.json
python scripts/run_contract.py data/contract/large_request.json -o out.json --labels data/contract/large_labels.json
```

`data/contract/large_request.json` собран из `data/large/truth/` скриптом `scripts/md_to_request.py`: 209 блоков, 2 источника, разметка на 42 блока.
