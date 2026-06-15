# Сравнение функций потерь обучения моделей эмбеддинга текстов

Тема домашнего задания: сравнить функции потерь, применяемые при обучении моделей эмбеддинга текстов, и проверить их влияние на качество retrieval / semantic similarity / classification экспериментов.

## Итоговые отчёты

- [Итоговый DOCX-отчет](./Горкунов%20Н.М.%20ИУ5-21М%20ММО%20ДЗ%20loss%20functions.docx)
- [Итоговый PDF-отчет](./render_check/report.pdf)

Интересующие функции потерь:
- Contrastive Loss: Hadsell, Chopra, LeCun, 2006, [Dimensionality Reduction by Learning an Invariant Mapping](https://ieeexplore.ieee.org/document/1640964)
- Triplet Loss: Hoffer, Ailon, 2014, [Deep Metric Learning using Triplet Network](https://arxiv.org/abs/1412.6622)
- InfoNCE Loss: van den Oord, Li, Vinyals, 2018, [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- NT-Xent Loss: Chen, Kornblith, Norouzi, Hinton, 2020, [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- SupCon Loss: Khosla et al., 2020, [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- Circle Loss: Sun et al., 2020, [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857)
- …

Дополнительные источники для текстовых эмбеддингов:
- Schroff, Kalenichenko, Philbin, 2015, [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) - важное применение triplet loss и online triplet mining.
- Reimers, Gurevych, 2019, [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- Gao, Yao, Chen, 2021, [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)

Окружение готовится через `uv`. Для PyTorch используется явный CUDA 12.8 index, как рекомендовано в документации uv для выбора PyTorch accelerator-specific wheels:

```powershell
uv sync
```

## Датасеты для сравнения

Основной эксперимент строится на английских датасетах. Все обучающие представления урезаются до `5000` строк `train` и `1000` строк `test` с фиксированным `seed=42`, чтобы разные функции потерь видели одинаковое количество обучающих anchor-записей.

Основные сценарии:
- `sentence-transformers/all-nli`: semantic relation / NLI, готовые pair и triplet представления.
- `sentence-transformers/quora-duplicates`: paraphrase / duplicate questions, готовые pair и triplet представления.
- `mteb/banking77`: короткие intent-запросы с 77 классами, основной датасет для SupCon и Circle Loss.
- `SetFit/20_newsgroups`: topic classification на более длинных текстах, второй class-label датасет для SupCon и Circle Loss.

Отдельная eval-only проверка:
- `sentence-transformers/stsb`: semantic textual similarity benchmark, не используется для обучения в основном сравнении.

Нормализованные представления:
- `contrastive_pair`: `anchor`, `other`, `label`.
- `triplet`: `anchor`, `positive`, `negative`, `label`.
- `positive_pair`: `anchor`, `positive`, `label` для InfoNCE / NT-Xent с in-batch negatives.
- `class_text`: `text`, `label`, `label_text` для SupCon / Circle Loss, только для class-label датасетов.

Команды подготовки:

```powershell
uv run python main.py inspect
uv run python main.py prepare
uv run python main.py verify
```

Результаты сохраняются в `data/processed`: JSONL-файлы по датасетам и представлениям, `manifest.json`, `summary.csv`, `dataset_inspection.csv`.
