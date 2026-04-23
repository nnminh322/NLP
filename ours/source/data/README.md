# Local Data Cache

This folder is the repository-local cache for T²-RAGBench snapshots.

Download the datasets once:

```bash
cd source
python data/download_t2ragbench.py --datasets all
```

The snapshots are saved to:

```text
source/data/t2-ragbench/FinQA
source/data/t2-ragbench/ConvFinQA
source/data/t2-ragbench/TAT-DQA
```

After that, training and evaluation code reads from disk only.

If you want to store the cache somewhere else, set:

```bash
export GSR_CACL_DATA_ROOT=/path/to/your/local/cache
```
