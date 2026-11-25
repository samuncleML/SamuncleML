# Multi-Head Plant Disease Classification for African Food Security

[![Framework](https://img.shields.io/badge/PyTorch-Training-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Deployment](https://img.shields.io/badge/TensorFlow_Lite-Edge_Deployment-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/lite)
[![Backbone](https://img.shields.io/badge/Backbone-MobileNetV3--Small-blue.svg)]()

> **Official Repository** for the research paper: *"Multi-Head Plant Disease Classification for African Food Security"* ([Placeholder for Conference/Journal Name & Link](https://zenodo.org/records/24535426)).

---

## Abstract & Overview

Sub-Saharan African agriculture faces a severe **diagnostic deficit**, where limited access to agricultural extension services leads to delayed or inaccurate identification of crop pathologies. This project introduces a lightweight, multi-task deep learning framework designed to democratize plant pathology diagnostics across rural farming communities. 

By utilizing a customized **Multi-Head MobileNetV3-Small** architecture, the system performs simultaneous crop species identification and disease classification directly on edge devices without requiring internet connectivity. This approach aims to safeguard yields and bolster food security for over **700 million people** across Africa who rely on staple and cash crops such as cassava, maize, and cocoa.

---

## Key Contributions & Features

* **Dual-Head Multi-Task Architecture:** A unified feature extraction backbone branches into two specialized classification heads, optimizing shared representations for:
  * **Head 1 (Species Identification):** 6 classes (5 target crops + 1 OOD rejection class).
  * **Head 2 (Pathology Classification):** 27 distinct disease and healthy states.
* **Edge-Optimized Footprint:** Built on **MobileNetV3-Small** leveraging Hardware-Aware Network Architecture Search (NAS) and `Hard-Swish` non-linearities, enabling real-time inference on low-cost smartphones.
* **Out-of-Distribution (OOD) Rejection:** Incorporates an explicit non-plant rejection class (filtering out soil, farming tools, hands, and background clutter) to mitigate false-positive diagnoses in uncontrolled field environments.
* **Offline-First Inference:** Quantized and converted via TensorFlow Lite (TFLite) for seamless deployment in low-connectivity rural regions.
* **Explainable AI (XAI) via Grad-CAM:** Integrates Gradient-weighted Class Activation Mapping to visualize symptomatic regions (e.g., necrotic lesions, chlorosis), fostering agronomic trust and model interpretability.

---

## Model Architecture & Performance

The network balances high diagnostic accuracy with strict computational constraints, requiring approximately $15\times$ fewer parameters than standard heavy backbones such as ResNet-101.

### Technical Specifications

| Specification | Details |
| :--- | :--- |
| **Backbone Network** | MobileNetV3-Small (Pre-trained & Fine-tuned) |
| **Total Parameters** | $\approx 2.5 \times 10^6$ (2.5 Million) |
| **Input Tensor Shape** | $224 \times 224 \times 3$ (RGB) |
| **Training Framework** | PyTorch |
| **Edge Runtime** | TensorFlow Lite (TFLite) |

### Quantitative Evaluation

Evaluated on a held-out test set emphasizing real-world field conditions, the dual-head model achieves state-of-the-art efficiency-to-accuracy trade-offs:

| Task Head | Number of Classes | Macro F1-Score |
| :--- | :---: | :---: |
| **Plant Species Identification** | 6 | **99.97%** |
| **Disease Classification** | 27 | **98.49%** |

---

## Dataset & Domain Adaptation

To bridge the **lab-to-field domain gap**—where models trained solely on uniform laboratory backgrounds fail under real agricultural conditions—our training pipeline utilizes a curated corpus of approximately **43,000 images** prioritized for in-field variance.

### Target Crops Covered
1. **Cassava** (*Manihot esculenta*)
2. **Corn / Maize** (*Zea mays*)
3. **Tomato** (*Solanum lycopersicum*)
4. **Cowpea** (*Vigna unguiculata*)
5. **Cocoa** (*Theobroma cacao*)
6. **Non-Plant / OOD** (Background & distractor objects)

### Augmentation Pipeline
To ensure invariance to illumination shifts, camera angles, and occlusion, the training pipeline applies stochastic transformations:
* Random geometric rotations and affine transformations
* Photometric color jittering (brightness, contrast, saturation)
* Horizontal and vertical flipping

---

## Repository Structure

```text
├── configs/                # Hyperparameters and model configuration files
├── data/                   # Dataset preprocessing and augmentation scripts
├── models/                 # Multi-Head MobileNetV3-Small PyTorch definitions
├── notebooks/              # Grad-CAM visualization and error analysis notebooks
├── export/                 # ONNX and TensorFlow Lite conversion utilities
├── train.py                # Multi-task training loop
├── evaluate.py             # Macro F1-score and confusion matrix evaluation
└── requirements.txt        # Python dependencies
```

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/your-username/multihead-plant-disease-africa.git
cd multihead-plant-disease-africa
pip install -r requirements.txt
```

### 2. Inference & Grad-CAM Visualization
```bash
python evaluate.py --weights checkpoints/mobilenetv3_multihead.pth --image sample_leaf.jpg --gradcam
```

### 3. Export to TensorFlow Lite
```bash
python export/convert_tflite.py --weights checkpoints/mobilenetv3_multihead.pth --output model_edge.tflite
```

---

## Future Work

* **Knowledge-Informed Machine Learning (KIML):** Integrating a *Biological Constraint Layer* into the output logits to mask biologically impossible host–pathogen pairs (e.g., preventing a cassava disease prediction on a maize leaf).
* **Multi-Input Multi-Output (MIMO) Fusion:** Incorporating multimodal tabular metadata—such as ambient humidity, temperature, soil pH, and geolocation—to refine disease probability priors.
* **Agronomic Prescription Engine:** Mapping calibrated confidence scores to localized, sustainable intervention protocols (organic and chemical treatment recommendations).

---

## Citation & References

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{author2026multihead,
  title={Multi-Head Plant Disease Classification for African Food Security},
  author={Author, First and Author, Second},
  booktitle={Proceedings of the International Conference on Computer Vision / Agricultural AI},
  year={2026}
}
```

### Foundational References
1. **Howard, A., et al.** (2019). *Searching for MobileNetV3*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324.
2. **IPCC** (2019). *Climate Change and Land: An IPCC Special Report on climate change, desertification, land degradation, sustainable land management, food security, and greenhouse gas fluxes in terrestrial ecosystems*.

---

## License
This project is released under the [MIT License](LICENSE).
