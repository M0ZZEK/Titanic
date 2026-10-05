# Titanic — Machine Learning Classification

Учебный ML-проект по задаче бинарной классификации
Kaggle Titanic: Machine Learning from Disaster.

Цель проекта — предсказать, выжил ли пассажир Titanic (`Survived`),
используя информацию о пассажире.

Проект охватывает полный ML lifecycle:
EDA, preprocessing, feature engineering, cross-validation,
подбор гиперпараметров, сравнение моделей, error analysis,
ансамблирование, нейронные сети и production inference.

## Data

Используются стандартные данные соревнования Titanic:

- `train.csv` — обучающая выборка;
- `test.csv` — тестовая выборка Kaggle;
- `gender_submission.csv` — пример submission.

Целевая переменная — `Survived`:

- `0` — пассажир не выжил;
- `1` — пассажир выжил.

## Validation strategy

Для оценки моделей использовались:

- Stratified 5-Fold Cross-Validation;
- Repeated Stratified Cross-Validation;
- Nested Cross-Validation;
- repeated holdout validation;
- GroupKFold по фамилиям;
- adversarial validation.

Основная метрика — `accuracy`.

Основной акцент сделан на локальную валидацию и устойчивость
результатов, а не на оптимизацию под Public Kaggle Leaderboard.

## Models

В проекте исследовались:

- Logistic Regression;
- K-Nearest Neighbors;
- Decision Tree;
- Random Forest;
- HistGradientBoosting;
- CatBoost;
- LightGBM;
- stacking;
- voting / averaging;
- Dense Neural Networks.

Для DNN дополнительно исследовались архитектура сети,
Batch Normalization, Dropout, activation functions, optimizers,
learning-rate scheduling, batch size и embeddings.

## Feature engineering

Проверялись:

- Title;
- FamilySize;
- IsChild;
- RelativeFare;
- TicketGroupSize;
- `log1p(Fare)`;
- interaction features.

Также проведены анализ пропусков и выбросов,
проверка constant / near-constant features,
корреляционный анализ, feature ablation и error analysis.

Новые признаки не добавлялись в production только ради
улучшения одного validation split.

## Final production model

Финальный production pipeline использует
`RandomForestClassifier`.

Основные параметры:

- `n_estimators=500`;
- `max_depth=8`;
- `min_samples_split=5`;
- `min_samples_leaf=1`;
- `max_features=None`;
- `random_state=43`.

Числовые признаки:

- Age;
- SibSp;
- Parch;
- Fare.

Preprocessing числовых признаков:

- median imputation;
- StandardScaler.

Категориальные признаки:

- Pclass;
- Sex;
- Embarked.

Preprocessing категориальных признаков:

- most-frequent imputation;
- OneHotEncoder с `handle_unknown='ignore'`.

Preprocessing и модель объединены в единый sklearn `Pipeline`.

## Validation result

Финальный Random Forest на фиксированном
Stratified 5-Fold CV:

`0.832766 ± 0.005663`

Дополнительные repeated и nested проверки использовались
для оценки устойчивости и контроля selection bias.

## Kaggle experiments

В ходе проекта были получены следующие Public Scores:

- `submission_rf_ticket.csv` — `0.77990`;
- `submission_from_main.csv` — исторический score `0.77751`;
- `submission_final_rf.csv` — `0.77272`;
- `submission_rf_holdout_80.csv` — `0.77272`.

Public Leaderboard использовался как дополнительная внешняя
проверка, а не как основной критерий выбора модели.

## Project structure

```text
Titanic/
├── Artifacts/
│   ├── titanic_rf_pipeline.joblib
│   └── submission_from_main.csv
├── Configs/
│   └── config.py
├── Data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   ├── submission_final_rf.csv
│   ├── submission_rf_holdout_80.csv
│   └── submission_rf_ticket.csv
├── Notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── inference.py
├── main.py
├── requirements.txt
└── README.md
```

## Environment

Проект проверялся на:

`Python 3.13.12`

Версии Python-пакетов зафиксированы в `requirements.txt`.

Установка зависимостей:

```bash
pip install -r requirements.txt
```

### LightGBM on macOS

Для LightGBM на macOS может потребоваться OpenMP runtime.

В использованном окружении:

```bash
conda install --override-channels -c conda-forge llvm-openmp -y
```

## Running inference

Из корневой директории проекта:

```bash
python main.py
```

`main.py`:

1. загружает `Data/raw/train.csv` и `Data/raw/test.csv`;
2. создаёт preprocessing + Random Forest pipeline;
3. оценивает модель с помощью Stratified 5-Fold Cross-Validation;
4. обучает pipeline на всей обучающей выборке;
5. сохраняет обученную модель в `Artifacts/titanic_rf_pipeline.joblib`;
6. выполняет prediction для Kaggle test;
7. создаёт `Artifacts/submission_from_main.csv`.

При текущей конфигурации воспроизводимый результат cross-validation:

`accuracy = 0.832766 ± 0.005663`.

Результат:

`Artifacts/submission_from_main.csv`

## Reproducibility

Для воспроизводимости:

- используется `random_state=43`;
- preprocessing находится внутри sklearn Pipeline;
- версии зависимостей зафиксированы;
- пути централизованы в `Configs/config.py`;
- production model сериализована через joblib;
- inference вынесен из notebook в отдельный модуль.

Production inference проверен отдельным запуском `main.py`.

Полученный submission полностью совпадает с
`Data/submission_final_rf.csv`:

`418 / 418 predictions`.

## Notes

При выборе финальной архитектуры основной акцент сделан на
устойчивость локальной валидации, отсутствие leakage,
контроль selection bias и воспроизводимость экспериментов.
