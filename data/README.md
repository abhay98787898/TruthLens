# Data

## Dataset: WELFake

The notebook requires the **WELFake Dataset** — a 72,134-article benchmark corpus introduced by Verma et al. (2021). It is not committed to this repository because the CSV is ~250 MB (above GitHub's 100 MB hard limit for non-LFS files).

## How to download

### Option 1 — Kaggle CLI (recommended)

If you have the [Kaggle CLI](https://www.kaggle.com/docs/api) configured with your API token:

```bash
kaggle datasets download -d saurabhshahane/fake-news-classification -p data/ --unzip
```

After extraction you should see:

```
data/WELFake_Dataset.csv
```

### Option 2 — Manual download

1. Go to https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
2. Click **Download** (you'll need a free Kaggle account)
3. Unzip the archive
4. Copy `WELFake_Dataset.csv` into this directory

## Expected file

| File | Rows × Cols | Size | Notes |
|---|---|---|---|
| `WELFake_Dataset.csv` | 72,134 × 4 | ~250 MB | Columns: index, `title`, `text`, `label` (0 = fake, 1 = real) |

## Working sample

By default the notebook uses a **stratified sample of 25,000 articles** (`SAMPLE_SIZE = 25000`, `RANDOM_STATE = 42` in the configuration cell) to keep runtime manageable. Learning curves in the analysis confirm that performance plateaus well before the full corpus size, so this is the sensible default.

To run on the full 72k, edit the configuration cell at the top of the notebook:

```python
SAMPLE_SIZE = None
```

## License and citation

The WELFake dataset is distributed on Kaggle for research use. When citing the dataset, please reference the original paper:

> Verma, P., Agrawal, P., Amorim, I., & Prodan, R. (2021). WELFake: Word embedding over linguistic features for fake news detection. *IEEE Transactions on Computational Social Systems*, 8(4), 881–893.
