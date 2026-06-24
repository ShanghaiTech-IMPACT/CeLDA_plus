<h1 align="center">CeLDA+: Prototypical Learning for Age-Robust Cephalometric Landmark Detection</h1>

<p align="center"><i>Official implementation &amp; the CephaAdoAdu46/201 benchmark</i></p>

<p align="center">
  <a href="https://hanwu.website/">Han Wu</a><sup>1,*</sup>,
  <a href="https://github.com/WeiJiaFiona">Wei Jia</a><sup>1,*</sup>,
  Lanzhuju Mei<sup>1</sup>,
  Tong Yang<sup>2</sup>,
  Min Zhu<sup>3</sup><br/>
  Haizhen Li<sup>3,†</sup>,
  <a href="https://cwangrun.github.io/">Chong Wang</a><sup>4,5,†</sup>,
  <a href="https://idea.bme.shanghaitech.edu.cn/">Dinggang Shen</a><sup>1,6,7,†</sup>,
  <a href="https://shanghaitech-impact.github.io/">Zhiming Cui</a><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup> ShanghaiTech University<br/>
  <sup>2</sup> Shanghai Linkedcare Information Technology<br/>
  <sup>3</sup> Shanghai Ninth People Hospital, SJTU<br/>
  <sup>4,5</sup> Stanford University<br/>
  <sup>6</sup> Shanghai United Imaging Intelligence<br/>
  <sup>7</sup> Shanghai Clinical Research and Trial Center
</p>

<p align="center">
  <sup>*</sup> Equal contribution &nbsp;&nbsp;&nbsp; <sup>†</sup> Corresponding authors
</p>

---

<p align="center">
  <img width="100%" alt="CeLDA+ pipeline" src="./figs/pipeline.png"/>
</p>

## Overview

**CeLDA+** (**Ce**phalometric **L**andmark **D**etection across **A**ges) is a **prototypical-learning** framework for **age-robust** cephalometric landmark detection. It reformulates landmark detection as **semantic matching** in a high-dimensional feature space — mapping morphologically diverse instances (e.g. unerupted teeth, mixed dentition) to compact regions around learnable prototypes — so a single model generalizes across adolescents and adults without any age-specific design. Two modules strengthen the prototypes:

- **Prototype Geometry Regularization** — enforces geometric consistency among landmarks.
- **Prototype Relation Mining** — captures semantic dependencies between anatomically related structures.

It supports the **46-** and **201-point** landmark sets and drives two downstream clinical analyses — **skeletal classification** and **cephalometric tracing** — alongside the released **CephaAdoAdu46/201** benchmark.

<details>
<summary><b>Abstract</b></summary>

Accurate cephalometric landmark detection is essential for orthodontic diagnosis. However, existing methods predominantly focus on adults and overlook adolescents, whose developmental variations, such as unerupted teeth and mixed dentition, introduce substantial appearance changes and degrade detection performance. A single framework that performs reliably across the full age spectrum within one unified model therefore remains lacking. To address this challenge, we propose **CeLDA+** (**Ce**phalometric **L**andmark **D**etection across **A**ges), a prototypical learning framework for age-robust landmark detection. Rather than relying on highly variable local appearances, it formulates landmark detection as semantic matching in a high-dimensional feature space. By mapping morphologically diverse instances to compact regions around their prototypes, the model learns age-robust representations without explicit age-specific designs. We further introduce two modules: **Prototype Geometry Regularization**, which enforces geometric consistency among landmarks, and **Prototype Relation Mining**, which captures semantic dependencies between anatomically related structures. For comprehensive evaluation, we construct two large-scale, multi-center datasets annotated with 46 and 201 landmarks, comprising 2,950 samples in total and representing the largest cephalometric benchmarks to date. Experiments on three benchmarks, including two age-diverse (mixed-age) and one adult-only dataset, show that CeLDA+ consistently outperforms prior state-of-the-art methods while maintaining the lowest computational cost. CeLDA+ also generalizes well to two downstream clinical evaluations: skeletal classification, following prior benchmark practice, and cephalometric tracing analysis, newly introduced in this work to provide contour-level evaluation for cephalometric landmark detection.

</details>

## Repository Structure

```text
code/
├── networks/           # CeLDA+ model + prototype relation mining
├── dataloaders/        # landmark dataset loading
├── utils/              # geometry, losses, soft-argmax, contour indexing
├── scripts/            # run_train / run_test / run_downstream
├── train.py            # training entry point
├── test.py             # evaluation / inference entry point
└── downstream_task.py  # skeletal classification + tracing (ASD/HD)
```

## Dataset

<p align="center">
  <img width="100%" alt="Landmark distribution" src="./figs/landmark_distribution1.png"/>
</p>

Our **CephaAdoAdu46/201** dataset comprises **2,950** multi-center lateral cephalograms of adolescent and adult patients, annotated with the **46-** and **201-point** protocols — the largest cephalometric landmark benchmark to date. It is available for **research purposes only**.

### Access

1. Visit the [IMPACT Lab dataset page](https://shanghaitech-impact.github.io/dataset/).
2. Download and complete the [application form](https://shanghaitech-impact.github.io/assets/dataset_application.pdf).
3. Send the **signed electronic copy** to [Han Wu](mailto:wuhan2022@shanghaitech.edu.cn) and [Dr. Zhiming Cui](mailto:cuizhm@shanghaitech.edu.cn), and **copy (CC) your advisor** as required in Section 5 of the form.
4. Access credentials (download link and password) will be provided once we receive the form.

### Data Organization

Place `CephaAdoAdu46/` and/or `CephaAdoAdu201/` under `data/`:

```text
data/CephaAdoAdu{46,201}
├── train/  (*.jpg + train_anno.json)
├── val/    (*.jpg + val_anno.json)
└── test/   (*.jpg + test_anno.json)
```

### Landmark Definitions

| Landmark Set | Definition File |
|:---:|:---:|
| 46 points | [`data/46pts_definition.json`](data/46pts_definition.json) |
| 201 points | [`data/201pts_definition.json`](data/201pts_definition.json) |

## Installation

```bash
git clone https://github.com/ShanghaiTech-IMPACT/CeLDA_plus.git
cd CeLDA_plus
conda create -n celdaplus python=3.8 -y
conda activate celdaplus
pip install -r requirements.txt
```

## Usage

The pipeline runs in three stages — **Training → Evaluation → Downstream Task** — driven by the scripts in [`code/scripts/`](code/scripts), each consuming the previous stage's output. Override defaults via environment variables (or run any entry point with `--help` for the full flag list):

| Variable | Stage | Default | Meaning |
|:---|:---|:---|:---|
| `GPU_ID` | all | `0` | CUDA device |
| `DATA_PATH` | train, eval | `data/CephaAdoAdu201` | Dataset root |
| `SAVE_PATH` | train | `workdir` | Output directory for runs |
| `EXP_NAME` | train | `CeLDA_201` | Experiment / run name |
| `RUN_DIR` | eval, downstream | *(required)* | Training output folder to consume |
| `CHECKPOINT` | eval, downstream | `best_ema` | Checkpoint to load (`best` / `best_ema`) |
| `EVAL_MODE` | downstream | `mix` | Prediction subset to analyze; must match the eval run |

The scripts target the **201-point** set; for the 46-point set, set `DATA_PATH=data/CephaAdoAdu46` and change `--number_of_keypoints` to `46` in the script.

### Training

```bash
bash code/scripts/run_train.sh
# override example: EXP_NAME=CeLDA_201_run2 GPU_ID=1 bash code/scripts/run_train.sh
```

Trains CeLDA+ on **CephaAdoAdu201** (100 epochs, batch size 4, base LR 1e-3) with the prototypical similarity objective plus a Wing coordinate loss, a cosine LR schedule with warmup, AMP, and EMA weights. Each run writes checkpoints (`best` / `best_ema`, selected by the lowest validation MRE), logs, and TensorBoard summaries under `SAVE_PATH/<EXP_NAME>/`.

### Evaluation

```bash
RUN_DIR=<your_training_output_dir> bash code/scripts/run_test.sh
```

Loads the `best_ema` checkpoint from `RUN_DIR` and decodes coordinates with Soft-Argmax, reporting per-landmark **MRE** (Mean Radial Error) and **SDR@1/2/3/4 mm** (Success Detection Rate) plus the overall mean error and inference efficiency. Pass `--eval_mode {adult,adolescent}` to `code/test.py` for the **age-robust** breakdown (default `mix`). Outputs are written to `RUN_DIR/eval_test_<checkpoint>_<eval_mode>/`:

- `metrics_table.txt` — per-landmark MRE / SDR table
- `test_prediction_results.json` — predictions per image (input to the downstream task)
- `efficiency_statistics.txt` — inference speed

### Downstream Task

```bash
RUN_DIR=<your_training_output_dir> bash code/scripts/run_downstream.sh
```

Consumes the evaluation's `test_prediction_results.json` (so `CHECKPOINT` / `EVAL_MODE` must match the eval run) and runs two clinical analyses:

1. **Skeletal classification** — derives standard cephalometric measurements from the predicted landmarks and reports the **Skeletal Classification Rate (SCR)**, overall and per class.
2. **Cephalometric tracing** — reconstructs anatomical contours and measures the contour-level **ASD** (Average Surface Distance) and **HD** (Hausdorff Distance) against the ground-truth tracing.

The landmark-to-contour mapping lives in [`code/utils/line_index.py`](code/utils/line_index.py); results (`bone_classification_table_*.txt`, `line_metrics.json`) are saved under `RUN_DIR/.../downstream_statistics/`.

## Roadmap

- [x] Repository creation & initial code commit
- [x] Training / evaluation code release
- [ ] Dataset release
- [ ] Paper and project page release

## Contact

For questions about the code or the dataset, please contact [Han Wu](mailto:wuhan2022@shanghaitech.edu.cn).
