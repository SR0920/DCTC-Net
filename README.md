# DCTC-Net: Dual Branch Cross-Fusion Transformer-CNN Architecture for Medical Image Segmentation

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/YOUR_ARXIV_ID_HERE) <!-- 论文发布后替换链接 -->
[![Conference](https://img.shields.io/badge/IEEE-TNNLS-blue)](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems) <!-- 如果有会议/期刊链接 -->
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository contains the official PyTorch implementation for the paper: **"DCTC-Net: Dual Branch Cross-Fusion Transformer-CNN Architecture for Medical Image Segmentation"**.

Our DCTC-Net is a novel hybrid architecture designed to leverage the complementary strengths of CNNs and Transformers for robust and efficient medical image segmentation. It features a **Dynamic Deformable Convolution (DDConv)** module to adaptively capture local features and a **(Shifted)-Window Adaptive Complementary Attention Module ((S)W-ACAM)** to model global dependencies with high efficiency.

<p align="center">
  <img src="assets/dctc_net_architecture.png" width="800" alt="DCTC-Net Architecture">
  <br>
  <em>Figure 1: The overall architecture of DCTC-Net.</em>
</p>

## 🌟 Highlights

- **Hybrid CNN-Transformer Design:** Achieves a superior balance of local detail extraction and global context modeling.
- **Novel Modules:** Introduces DDConv and (S)W-ACAM to specifically address challenges in medical imaging, such as organ deformation and complex backgrounds.
- **High Efficiency:** Delivers state-of-the-art performance with fewer parameters and lower computational cost compared to many existing methods.
- **No Pre-training Required:** Achieves excellent results without relying on pre-training on large-scale datasets like ImageNet.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/DCTC-Net.git
    cd DCTC-Net
    ```

2.  **Create a conda environment (Recommended):**
    ```bash
    conda create -n dctcnet python=3.8 -y
    conda activate dctcnet
    ```

3.  **Install dependencies:**
    Our code is built on PyTorch. Please install the required packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include:
    - Python 3.8+
    - PyTorch 1.7.0+
    - TorchVision
    - SimpleITK
    - NiBabel
    - tqdm
    - ... (列出其他关键依赖)

## 📊 Datasets

In our paper, we evaluated DCTC-Net on three public datasets:

1.  **ISIC 2018:** [Link to dataset or download instructions]
2.  **LiTS-Liver (LiTS17):** [Link to dataset or download instructions]
3.  **ACDC:** [Link to dataset or download instructions]

**Data Preparation:**

Please follow these steps to prepare the data. We expect the directory structure to be as follows:

```
./data/
├── ISIC2018/
│   ├── images/
│   └── masks/
├── LiTS/
│   ├── images/
│   └── masks/
├── ACDC/
│   ├── images/
│   └── masks/
```

You can run the preprocessing script to convert the data into the required format (e.g., `.npy` files).

```bash
python preprocess/prepare_isic.py --data_path ./data/ISIC2018
python preprocess/prepare_lits.py --data_path ./data/LiTS
# ... etc.
```
*(请在此处提供更详细的数据预处理说明，或者链接到一个单独的 `data_preparation.md` 文件)*

## Training

To train the DCTC-Net model on a specific dataset, use the `train.py` script.

**Example for training DCTC-Net-T on the ISIC 2018 dataset:**

```bash
python train.py --dataset ISIC2018 --model_name dctc_net_t --batch_size 16 --epochs 200 --lr 0.001
```

**Key Arguments:**

-   `--dataset`: Name of the dataset to use (e.g., `ISIC2018`, `LiTS`, `ACDC`).
-   `--model_name`: The model variant to train (`dctc_net_t` or `dctc_net_b`).
-   `--batch_size`: Training batch size.
-   `--epochs`: Total number of training epochs.
-   `--lr`: Initial learning rate.

Training logs and model checkpoints will be saved to the `experiments/` directory by default.

## 🧪 Evaluation

To evaluate a trained model, use the `test.py` script. You need to provide the path to the trained model checkpoint.

**Example for evaluating DCTC-Net-T on the ISIC 2018 test set:**

```bash
python test.py --dataset ISIC2018 --model_name dctc_net_t --checkpoint_path experiments/ISIC2018_dctc_net_t/best_model.pth
```

The script will output the evaluation metrics (Dice, JA, SE, etc.) reported in the paper.

## 🚀 Pre-trained Models

We provide the pre-trained weights for our DCTC-Net-T and DCTC-Net-B models for all three datasets to facilitate reproducibility.

| Model       | Dataset    | Download Link                                 |
|-------------|------------|-----------------------------------------------|
| DCTC-Net-T  | ISIC 2018  | [Link to .pth file]                           |
| DCTC-Net-B  | ISIC 2018  | [Link to .pth file]                           |
| DCTC-Net-T  | LiTS-Liver | [Link to .pth file]                           |
| ...         | ...        | ...                                           |

You can download them and use the `--resume` flag in `test.py` to load the weights.

## 🌟 Results

Our DCTC-Net achieves state-of-the-art performance on multiple medical image segmentation benchmarks.

**(Optional: 在这里可以插入一个关键的结果表格或图表)**

| Model      | Dataset    | Dice (%) | JA (%)  |
|------------|------------|----------|---------|
| DCTC-Net-B | ISIC 2018  | **91.23**| **84.76** |
| DCTC-Net-B | LiTS-Liver | **96.82**| ...     |
| DCTC-Net-B | ACDC       | **91.95**| ...     |

For more details, please refer to our paper.

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{YourName2024DCTCNet,
  title={DCTC-Net: Dual Branch Cross-Fusion Transformer-CNN Architecture for Medical Image Segmentation},
  author={Author 1 and Author 2 and ...},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  % volume={XX},
  % number={XX},
  % pages={XXXX-XXXX},
}
```

## Acknowledgements

We would like to thank ... (感谢任何提供帮助的个人或组织，例如数据集提供方、计算资源支持等).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.# DCTC-Net
