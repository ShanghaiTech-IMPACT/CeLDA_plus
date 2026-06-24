<h1 align="center">CeLDA+ & CephaAdoAdu Dataset</h1>

<p align="center">
  <a href="https://visitor-badge.laobi.icu/badge?page_id=ShanghaiTech-IMPACT.CeLDA_plus">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=ShanghaiTech-IMPACT.CeLDA_plus" alt="visitors" />
  </a>
  <a href="https://github.com/ShanghaiTech-IMPACT/CeLDA_plus/stargazers">
    <img src="https://img.shields.io/github/stars/ShanghaiTech-IMPACT/CeLDA_plus?style=social" alt="GitHub Repo stars" />
  </a>
</p>

<p align="center">
  <img width="50%" alt="CeLDA+ logo" src="./figs/logo.png"/>
</p>



<p align="center">
  <a href="#">[Paper]</a> &nbsp;|&nbsp;
  <a href="https://shanghaitech-impact.github.io/dataset/">[Dataset]</a> &nbsp;|&nbsp;
  <a href="#">[Project Page]</a>
</p>

---

## 🚧 TODO

- [x] Repository creation & initial code commit
- [x] Training / evaluation code release
- [ ] Dataset page online
- [ ] Dataset release
- [ ] Paper and project page release



## 📝 Paper

> **CeLDA+: Prototypical Learning for Age-Robust Cephalometric Landmark Detection**
>
> [Han Wu](https://hanwu.website/)<sup>1*</sup>, [Wei Jia](https://github.com/WeiJiaFiona)<sup>1*</sup>, Lanzhuju Mei<sup>1</sup>, Tong Yang<sup>2</sup>, Min Zhu<sup>3</sup>, Haizhen Li<sup>3✉</sup>, [Chong Wang](https://cwangrun.github.io/)<sup>4,5✉</sup>, [Dinggang Shen](https://idea.bme.shanghaitech.edu.cn/)<sup>1,6,7✉</sup>, [Zhiming Cui](https://shanghaitech-impact.github.io/)<sup>1✉</sup> <br/>
> <sup>1</sup> School of Biomedical Engineering & State Key Laboratory of Advanced Medical Materials and Devices, ShanghaiTech University, Shanghai, China <br/>
> <sup>2</sup> Shanghai Linkedcare Information Technology Co., Ltd., Shanghai, China <br/>
> <sup>3</sup> Shanghai Ninth People Hospital, School of Medicine, Shanghai Jiao Tong University, Shanghai, China <br/>
> <sup>4</sup> Center for Artificial Intelligence in Medicine and Imaging, Stanford University, Palo Alto, CA, USA <br/>
> <sup>5</sup> Department of Radiology, Stanford University, Stanford, CA, USA <br/>
> <sup>6</sup> Shanghai United Imaging Intelligence Co. Ltd., Shanghai, China <br/>
> <sup>7</sup> Shanghai Clinical Research and Trial Center, Shanghai, China <br/>
> <sup>*</sup> These authors contributed equally to this paper. ✉ Corresponding authors: [Haizhen Li](mailto:lihaizhen_dent@163.com), [Chong Wang](mailto:chongwa@stanford.edu), [Dinggang Shen](mailto:Dinggang.Shen@gmail.com), [Zhiming Cui](mailto:cuizhm@shanghaitech.edu.cn).


## 📚 Dataset
<p align="center">
  <img width="100%" alt="Landmark distribution" src="./figs/landmark_distribution1.png"/>
</p>
Our **CephaAdoAdu46/201** dataset is available for **research purposes only**.

### Access

1. Visit the [IMPACT Lab dataset page](https://shanghaitech-impact.github.io/dataset/).
2. Download and fill out the [application form](https://shanghaitech-impact.github.io/assets/dataset_application.pdf).
3. Send the **signed e-copy** to [Han Wu](mailto:wuhan2022@shanghaitech.edu.cn) and [Dr. Zhiming Cui](mailto:cuizhm@shanghaitech.edu.cn), **CC your advisor** as required in Sec. 5 of the form.
4. We will send the dataset link and password upon receiving the form.

### Data Organization

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



## 💻 Getting Started

### Installation

```bash
git clone https://github.com/ShanghaiTech-IMPACT/CeLDA_plus.git
cd CeLDA_plus
conda create -n celdaplus python=3.8 -y
conda activate celdaplus
pip install -r requirements.txt
```

### Data

Organize the dataset as described in [Data Organization](#data-organization) above (place `CephaAdoAdu46/` and/or `CephaAdoAdu201/` under `data/`). Landmark definitions are provided in [`data/46pts_definition.json`](data/46pts_definition.json) and [`data/201pts_definition.json`](data/201pts_definition.json).

### Training

```bash
bash code/scripts/run_train.sh
```

Options can be set via environment variables (or by editing the script / calling `code/train.py` directly, see `--help`): `DATA_PATH`, `SAVE_PATH`, `EXP_NAME`, `GPU_ID`.

### Testing / Inference

```bash
RUN_DIR=<your_training_output_dir> bash code/scripts/run_test.sh
```

### Downstream: skeletal classification & cephalometric tracing

```bash
RUN_DIR=<your_training_output_dir> bash code/scripts/run_downstream.sh
```

This evaluates skeletal classification and the contour-level **cephalometric tracing analysis**; the landmark-to-contour mapping is defined in [`code/utils/line_index.py`](code/utils/line_index.py) and the contour-level ASD/HD computation in [`code/downstream_task.py`](code/downstream_task.py).

### Repository Structure

```text
code/
├── networks/           # CeLDA+ model (CeLDA_Plus.py); Prototype Relation Mining (Masked_Modeling.py)
├── dataloaders/        # landmark dataset loading
├── utils/              # geometry, losses, soft-argmax, line/contour indexing, helpers
├── scripts/            # run_train.sh / run_test.sh / run_downstream.sh
├── train.py            # training entry point
├── test.py             # inference / evaluation entry point
└── downstream_task.py  # skeletal classification + cephalometric tracing (ASD/HD) evaluation
```
