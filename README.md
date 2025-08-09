# 🩺 TL-S-CycleGAN for Tumor Classification in Medical Imaging

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Open%20in-Colab-yellow.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of my research paper:  
**_"Exploring the Impact of Transfer Learning and Semi-Supervised GAN for Tumor Classification in Medical Imaging"_**

---

## 📖 About This Project

This research was carried out as my **final year project** during the **8th semester of BSCS**.  
It was my **first computer vision project in medical AI**, focusing on breast cancer diagnosis from histopathology images using deep learning.

The work addresses **data scarcity** and **class imbalance** in medical datasets by generating realistic synthetic tumor images through **Generative Adversarial Networks (GANs)** combined with **transfer learning**.

I developed **two novel TL-S-CycleGAN variants**:
- **TL-S-CycleGAN (ResNet-50 discriminator)**
- **TL-S-CycleGAN (VGG-16 discriminator)**

For comparison, I also implemented a **baseline Simple CycleGAN** based on the original work of **Jun-Yan Zhu et al. (2017)**.

---

## 🏆 Key Achievements

- Designed and implemented **two original TL-S-CycleGAN variants** with transfer learning–based discriminators.
- Improved classification accuracy to **95%** using TL-S-CycleGAN (ResNet-50).
- Achieved the highest image quality (SSIM, PSNR) with TL-S-CycleGAN (VGG-16).
- Boosted segmentation IoU score to **0.6787**, outperforming the baseline CycleGAN.
- Demonstrated that **transfer learning + GAN augmentation** can match or surpass state-of-the-art results on the BreakHis dataset.

---

## 📂 Dataset

We use the **BreakHis Breast Cancer Histopathology Dataset**:  
🔗 [BreakHis on Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)

**Details:**
- 7,909 microscopic biopsy images from 82 patients
- Magnifications: 40×, 100×, 200×, 400×
- 2,480 benign | 5,429 malignant

---

## 🚀 Run in Google Colab

### **Training Notebooks**
| Model | Colab Link |
|-------|------------|
| Simple CycleGAN (Jun-Yan Zhu et al., 2017) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Training/training_simple_cyclegan.ipynb) |
| TL-S-CycleGAN (ResNet-50) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Training/training_resnet_50_cyclegan.ipynb) |
| TL-S-CycleGAN (VGG-16) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Training/training_vgg_16_model.ipynb) |

---

### **Testing Notebooks**
| Model | Colab Link |
|-------|------------|
| Simple CycleGAN (Jun-Yan Zhu et al., 2017) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Testing/testing_simple_cyclegan_trained_model_generating_images.ipynb) |
| TL-S-CycleGAN (ResNet-50) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Testing/testing_resnet_50_trained_model_generating_images.ipynb) |
| TL-S-CycleGAN (VGG-16) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Testing/testing_vgg_16_model_generating_images.ipynb) |

---

### **Metrics & Visualizations**
| File | Description | Colab Link |
|------|-------------|------------|
| `Classification_Performance_and_FCN_Metrics_Visuals.ipynb` | Accuracy, F1-score, precision, recall + FCN visuals for all CycleGAN variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Evaluation%20Metrics%20and%20Visuals/Classification_Performance_and_FCM_metrics_Visuals.ipynb) |
| `Evaluation_Metrics_Image_Quality.ipynb` | SSIM, PSNR, MSE calculations + visual comparisons | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/TL-S-CycleGAN-Histopathology/blob/main/Evaluation%20Metrics%20and%20Visuals/Evaluation_Metrics_Image_Quality.ipynb) |

---

## 📊 Sample Generated Images

Below are example image translations produced by the three models for both directions: **Benign → Malignant** and **Malignant → Benign**.

### 🔹 Simple CycleGAN (Jun-Yan Zhu et al., 2017)
**Benign → Malignant**
![](Generated%20Images/Simple_Cyclegan/benign_to_malignant/0.png)

**Malignant → Benign**
![](Generated%20Images/Simple_Cyclegan/malignant_to_benign/0.png)

---

### 🔹 TL-S-CycleGAN (ResNet-50)
**Benign → Malignant**
![](Generated%20Images/Resnet_50_Cyclegan/benign_to_malignant/0.png)

**Malignant → Benign**
![](Generated%20Images/Resnet_50_Cyclegan/malignant_to_benign/0.png)

---

### 🔹 TL-S-CycleGAN (VGG-16)
**Benign → Malignant**
![](Generated%20Images/VGG_16_Cyclegan/benign_to_malignant/0.png)

**Malignant → Benign**
![](Generated%20Images/VGG_16_Cyclegan/malignant_to_benign/0.png)


---

## 📊 Summary of Results

- **Image Quality:** TL-S-CycleGAN (VGG-16) → Best SSIM & PSNR, most realistic outputs.  
- **Classification:** TL-S-CycleGAN (ResNet-50) → Highest accuracy & F1-score.  
- **Segmentation:** TL-S-CycleGAN (VGG-16) → Best IoU & per-class accuracy.  

---

## 📽️ Project Presentation
For a quick overview of the project, models, methodology, and results, check out the presentation below:

- [View PDF Presentation](Presentation/FYP_research_Hussam.pdf)
- [Download PPTX Version](Presentation/FYP_research_Hussam.pptx)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> **Note:**  
> Due to large `.ipynb` file sizes, outputs have been cleared from **training notebooks** in this repository.  
> If you require access to the original notebooks with full outputs, please email me at **hussamumer28092000@gmail.com** and I will provide you with a Colab link.  
> All notebooks are stored directly in the repository root for easier access via Google Colab.

