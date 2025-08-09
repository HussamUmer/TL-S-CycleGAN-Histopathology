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
| `classification_metrics_fcn_visuals.ipynb` | Accuracy, F1-score, precision, recall + FCN visuals for all CycleGAN variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/repo/blob/main/classification_metrics_fcn_visuals.ipynb) |
| `image_quality_metrics.ipynb` | SSIM, PSNR, MSE calculations + visual comparisons | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/repo/blob/main/image_quality_metrics.ipynb) |

---

## 📊 Summary of Results

- **Image Quality:** TL-S-CycleGAN (VGG-16) → Best SSIM & PSNR, most realistic outputs.  
- **Classification:** TL-S-CycleGAN (ResNet-50) → Highest accuracy & F1-score.  
- **Segmentation:** TL-S-CycleGAN (VGG-16) → Best IoU & per-class accuracy.  

---

## 📜 Citation

**CycleGAN Original Paper:**
