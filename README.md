# Text-to-SQL — QLoRA Fine-tuned Qwen2.5-1.5B

Turns a plain-English question + a table schema into a SQL query. Fine-tuned, not prompted — a QLoRA adapter trained on top of an open-source base model.

**Live demo:** _add your Space or share link here_

## Results

| | Execution accuracy |
|---|---|
| Zero-shot (base model) | 85.4% |
| Fine-tuned (this adapter) | **90.9%** (+5.6 pts) |

Measured on 200 held-out test examples never seen during training. "Execution accuracy" means the generated SQL is run against sample data and its result is compared to the gold query's result — not just string-matched.

## Stack
- **Base model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Fine-tuning:** QLoRA (4-bit) via `peft` + `trl`'s `SFTTrainer`, 2 epochs
- **Dataset:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78k schema + question + SQL examples, 10k subset used
- **Adapter:** [Harsh3567475586/qwen2.5-1.5b-text-to-sql-lora](https://huggingface.co/Harsh3567475586/qwen2.5-1.5b-text-to-sql-lora) on HF Hub
- **Demo:** Gradio

## Repo structure
| File | What it does |
|---|---|
| `01_data_prep.ipynb` | Load, format, and split the dataset |
| `02_finetune_and_evaluate.ipynb` | QLoRA fine-tuning + zero-shot vs. fine-tuned evaluation |
| `03_push_to_hub.ipynb` | Push the trained adapter to HF Hub |
| `app.py` + `requirements.txt` | Gradio Space app (standard deployment target) |
| `app_colab_demo.py` | Same app, run from Colab with a public share link — a free workaround since HF Spaces currently gates the Gradio SDK behind PRO for new accounts |

## Reproducing this
1. **Data prep** (`01_data_prep.ipynb`) — Colab, CPU is enough. Downloads a zip of the processed splits.
2. **Fine-tune + evaluate** (`02_finetune_and_evaluate.ipynb`) — Kaggle, GPU. Designed to run via **Save & Run All** (background job) rather than an interactive session, so it isn't lost if the tab closes or the session idles out mid-training. Upload the Phase 1 zip as a Kaggle Dataset first.
3. **Push to Hub** (`03_push_to_hub.ipynb`) — Colab. Needs a write-scoped HF token.
4. **Demo** — deploy `app.py` to an HF Space (SDK: Gradio), or run `app_colab_demo.py` from Colab for a temporary public link.

## Notes on the eval methodology
`sql-create-context` ships table schemas but no real rows, so execution accuracy is measured against **synthetic data** generated to match each schema's column types. This means multi-table `JOIN` queries are noisier than single-table ones, since random values don't preserve real foreign-key relationships — a known, accepted limitation for this dataset rather than a bug in the eval.
