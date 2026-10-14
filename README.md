# Lung Cancer Detection from CT Images (ResNet-50, PyTorch)

A transfer-learning project that classifies lung CT images into **cancerous** and **non-cancerous** categories using a pretrained ResNet-50 convolutional neural network (CNN).

Developed for **COMP1827 Introduction to Artificial Intelligence**, University of Greenwich (2025/26), as the individual implementation component of a group project.

> **Educational research only.** This model has not been clinically validated and must not be used to diagnose or rule out cancer.

## Methodology

1. Reviewed research on CT-based lung cancer classification and selected a pretrained ResNet-50 architecture.
2. Excluded the benign category to focus on binary classification.
3. Froze the pretrained layers and trained the final classification layer for **5 epochs**.
4. Unfroze the final convolutional block and fine-tuned the model for **5 additional epochs**.
5. Used the **Adam optimiser**, a **0.0001 initial learning rate** (**0.00001 for fine-tuning**), **batch size 32**, and **class weighting** to account for the two categories.
6. Exchanged **50 unique images** between the training and testing collections to introduce CT images from different sources, as described in the project report.
7. Evaluated predictions with a confusion matrix and classification report. The uploaded training script generates these metrics for its own test split; the paper's reported results are documented separately below.

## Code origin and my experiment

The training implementation is **adapted from Seyed Vala Khorasani's Kaggle ResNet-50 notebook**; it is not presented as an original model implementation. For my coursework, I selected/prepared CT datasets, ran training and evaluation, and analysed the outputs. The separate `predict.py` script loads the saved model for individual-image predictions. The precise line-by-line differences from the source notebook have not been independently verified.

## Datasets and preparation

Three publicly available datasets were used. The source counts below describe the **dataset preparation reported in the paper**, not the verified composition of the final evaluated subset.

| Role | Dataset | Reported images | Source |
|---|---|---|---|
| Training source | IQ-OTH/NCCD (Aditya Mahimkar, 2021) | 1,201 cancerous; 1,208 non-cancerous | [Kaggle](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) |
| Testing source | CT-Scan Images (Orvile, 2021) | 238 cancerous; 120 non-cancerous | [Kaggle](https://www.kaggle.com/datasets/orvile/ct-scan-images) |
| Additional normal images | Lungcancer4types-imagedataset (Kabil007, 2021) | 6 non-cancerous | [Kaggle](https://www.kaggle.com/datasets/kabil007/lungcancer4types-imagedataset) |

The report describes **2,409 training images** and a prepared testing collection of **364 images** (238 cancerous and 126 non-cancerous). However, the published classification report covers a different distribution of **362 evaluated images** (180 cancerous and 182 non-cancerous). **The reason for this discrepancy has not yet been established from the evaluation code.** The reported evaluation metrics below therefore apply **only to those 362 images**; they should not be interpreted as results on all 364 images described in the dataset preparation.

Datasets are not redistributed in this repository. Please consult the original sources for access and licensing terms.

**Important:** `train.py` currently reads images from a local folder named `Augmented IQ-OTHNCCD lung cancer dataset` and makes a **stratified 70% train / 15% validation / 15% test split** of the images found there. This is **not the same as** the separately sourced train/test arrangement described in the paper. The script cannot, on its own, substantiate the paper's 362-image classification report. The model checkpoint and dataset contents used in that run are not included.

## Reported results

The classification report in the project paper gives the following results for **362 evaluated CT images**:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-cancerous | 0.98 | 1.00 | 0.99 | 182 |
| Cancerous | 1.00 | 0.98 | 0.99 | 180 |
| **Overall accuracy** | | | **0.99** | **362** |

**Accuracy reported in the coursework paper: 99% (not independently reproduced from the uploaded scripts).** Cancerous-class recall of 0.98 indicates that some cancerous images were missed; false negatives are particularly important in medical applications. These are **reported experimental metrics**, not a measure of clinical diagnostic reliability or performance on unseen hospitals' data.

## Limitations and future work

- **Evaluation reproducibility:** Recheck the dataset-loading and evaluation code to explain the 364-versus-362 difference and the changed class distribution.
- **Generalisation:** Training images predominantly came from one dataset, so results may not transfer to scans from other hospitals, scanners, or patient populations.
- **Potential leakage:** Verify that the 50-image exchange did not create overlapping images or patient-level overlap between training and evaluation sets.
- **Interpretability:** Add Grad-CAM or similar visual explanation methods.
- **Further evaluation:** Test on independently held-out, patient-separated datasets and consider multi-class classification, including benign cases.

## Attribution and credits

- **Original code and licence:** This project adapts Valakhorasani (2021), [*Lung Cancer Detection with Pretrained ResNet-50*](https://www.kaggle.com/code/valakhorasani/lung-cancer-detection-with-pretrained-resnet-50/notebook), published under the **Apache License 2.0**. The original author and source are credited. When publishing adapted code, include a copy of the [Apache 2.0 licence](https://www.apache.org/licenses/LICENSE-2.0), retain applicable copyright and NOTICE information, and identify modifications.
- **My coursework activity:** selection and preparation of datasets from three sources, reported exchange of 50 images, running and analysing transfer-learning/fine-tuning experiments, and examining predictions and confusion-matrix results. The training code itself is adapted from the credited notebook; specific code-level modifications have not been independently established.
- **Architecture:** ResNet-50, introduced by He et al. (2016), with pretrained weights through `torchvision.models`.
- **Tools reported for the project:** PyTorch, torchvision, OpenCV, and Pillow.
- **Datasets:** Credit belongs to the respective Kaggle dataset publishers linked above.

## References

- He, K., Zhang, X., Ren, S. and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR, pp. 770–778.
- Arun, M. et al. (2024). *Lung Cancer Detection Using ResNet-50 CNN Architecture*. ICAIT, vol. 1, pp. 1–6.
- Tasmim et al. (2023). *A Lightweight-CNN Model for Efficient Lung Cancer Detection and Grad-CAM Visualization*. ICICT4SD, pp. 254–258.
- Tulasi, R. and Jadala, V. C. (2025). *Optimized ResNet and EfficientNet Architectures for Accurate and Efficient Lung Cancer Detection in CT Scans*. ICC-ROBINS, pp. 682–686.
- Wang, L. (2022). *Deep Learning Techniques to Diagnose Lung Cancer*. Cancers, 14(22), 5569.

## Running the project

1. Download the datasets from the linked Kaggle pages, subject to their licences. **No CT images are included here.**
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure `dataset_root` in `train.py` to point to your locally prepared dataset folder with `Malignant cases` and `Normal cases` subfolders.
4. Run `python train.py` to train, evaluate and save `resnet50_lung_cancer_binary_detector_training.pth`. Results depend on the dataset actually supplied, random split, software versions, and training environment.
5. To run individual-image predictions, set `ROOT_FOLDER` and `MODEL_PATH` in `predict.py` and run `python predict.py`. The expected subfolders are `Cancerous raw images - jpg` and `Non-Cancerous raw images - jpg`.

The `.pth` model weights are not included; you must generate them through training or provide your own compatible checkpoint. The reported paper results have **not** been verified by rerunning these scripts in this repository. This is a research/educational demonstration, not a medical diagnostic tool.
