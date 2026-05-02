# РК1: target encoding и диаграмма рассеяния

Горкунов Николай Максимович ИУ5-21М

| Номер варианта | Номер задачи №1 | Номер задачи №2 |
| ----- | ----- | ----- |
| 2 | 2 | 22 |

Дополнительные требования по группам:

- Для студентов групп ИУ5-21М, ИУ5И-21М - для пары произвольных колонок данных построить график "Диаграмма рассеяния".

### Задача №2

Для набора данных проведите кодирование одного (произвольного) категориального признака с использованием метода "target (mean) encoding".

Задание выполнено на датасете
[CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles).

Что сделано:

- загружен CSV-файл `data/CO2_Emissions_Canada.csv`;
- для категориального признака `Make` выполнено target / mean encoding;
- целевой признак: `CO2 Emissions(g/km)`;
- добавлен новый признак `Make_mean_encoded`;
- построена диаграмма рассеяния для `Engine Size(L)` и `CO2 Emissions(g/km)`.

Запуск:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py
```

Сохраненные результаты:

- `results/co2_emissions_with_mean_encoding.csv` - исходные данные с новой encoded-колонкой;
- `results/mean_encoding_by_make.csv` - среднее значение целевого признака для каждой марки;
- `results/scatter_engine_size_co2.png` - диаграмма рассеяния;
- `results/summary.txt` - краткий текстовый отчет.
