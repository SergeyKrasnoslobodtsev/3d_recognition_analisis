# Дорожная карта

## Этап 1 — подготовка данных

- STEP→B-Rep парсер (OpenCascade/ocp) + ваш пайплайн извлечения: UV-растр для каждой грани (фиксированный размер, напр. 128×128), граф смежности, нормализация масштаба. 
- Генерация аугментаций на уровне B-Rep (отключаем «ломающие» топологию операции).

- Предрасчёт «само-меток» (surface/curve types, габариты) — пригодится для auxiliary losses.

**побочный материал для этапа**

1) [UV-Net: Learning from Boundary Representations](https://openaccess.thecvf.com/content/CVPR2021/papers/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.pdf?utm_source=chatgpt.com), 


2) [Self-Supervised Representation Learning for CAD](https://openaccess.thecvf.com/content/CVPR2023/papers/Jones_Self-Supervised_Representation_Learning_for_CAD_CVPR_2023_paper.pdf?utm_source=chatgpt.com)

3) [Реализация статьи на GitHub Self-Supervised Representation Learning for CAD](https://github.com/zhenshihaowanlee/Self-supervised-BRep-learning-for-CAD/blob/main/Example_data/sampled_data.py)



## Этап 2 — SSL-предобучение

- Энкодер в духе UV-Net (image CNN на UV + GCN/GAT на графе) → pooling → эмбеддинг модели (d=256/512).

- Loss: InfoNCE с температурой; batch-wise hard negatives (введите simple miner по габаритам/объёму, чтобы не учиться на тривиальных диссимациях).

- Валидируем на proxy-метриках: Recall@k в «каталожных» группах, где вы сами создадите пары «оригинал ↔ модификация» (скругления/фаски/отверстия).

## Этап 3 — Индекс и API

- FAISS-индекс + сервис выдачи топ-k.

- Реренкинг: усреднение эмбеддингов граней, затем ML-score с учётом совпадения функции симметрий/отношений размеров (feature-based rerank). 

## Этап 4 — Подсветка отличий

- Выравнивание CAD↔CAD (feature-based ICP на ориентированных гранях).

- Поиск соответствий граней (эмбеддинг граней + топоконтекст) и рендер heatmap «что добавлено/удалено/изменено».

## Этап 5 — Текст/фильтры (MVP+)

- Псевдокорпус описаний из B-Rep-атрибутов → контрастивная привязка текста к эмбеддингам моделей.

- UI-фильтры поверх предрасчитанных признаков (число отверстий, радиусы, bounding box и пр.).