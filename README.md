<<<<<<< HEAD
## Hi there 👋

<!--
**samuncleML/SamuncleML** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...# Multi-Head Plant Disease Classification for African Food Security

## Overview
This project presents a specialized deep learning solution designed to address the **diagnostic deficit** in African agriculture. By leveraging a custom **Multi-Head MobileNetV3-Small** architecture, the system enables **offline, high-precision identification** of crop species and their associated diseases.

The goal is to strengthen food security for over **700 million Africans** who depend on staple crops such as **cassava** and **corn**.

---

## Key Features

- **Dual-Head Architecture**  
  A shared backbone performs simultaneous:
  - Plant Identification (6 classes)
  - Disease Classification (27 classes)

- **Mobile-First Design**  
  Optimized for low-resource devices using MobileNetV3-Small with:
  - Hardware-Aware NAS  
  - Hard-Swish activation

- **Out-of-Distribution (OOD) Rejection**  
  Dedicated class to reject non-plant inputs (e.g., soil, tools, hands), preventing invalid predictions.

- **Offline Accessibility**  
  Deployed via an offline API to ensure usability in rural regions with limited or no internet connectivity.

- **Interpretability**  
  Grad-CAM visualizations highlight disease-relevant regions (lesions, chlorosis), ensuring transparency and trust.

---

## Technical Specifications

- **Backbone:** MobileNetV3-Small  
- **Parameters:** ~2.5 million (≈15× fewer than ResNet-101)  
- **Input Size:** `224 × 224 × 3`  
- **Frameworks:**  
  - PyTorch (Training)  
  - TensorFlow Lite (Deployment)

### Accuracy Results
- **Plant Identification:** 99.97% Macro F1-score  
- **Disease Classification:** 98.49% Macro F1-score  

---

## Dataset

The model was trained on approximately **43,000 images**.  
To bridge the **lab-to-field gap**, the dataset prioritizes real-world farm images and applies extensive data augmentation, including:

- Rotation  
- Color jittering  
- Horizontal and vertical flips  

This improves robustness against inconsistent lighting and background noise.

### Target Crops
- Corn (Maize)
- Cassava
- Tomato
- Cowpea
- Cocoa

---

## Future Directions

- **KIML Integration**  
  Introduce a Biological Constraint Layer to eliminate impossible plant–disease combinations.

- **MIMO Expansion**  
  Incorporate environmental metadata such as humidity, temperature, and soil pH.

- **Prescription Engine**  
  Use confidence scores to recommend targeted organic or chemical treatments.

---

## References

- IPCC, *Climate Change and Land: An IPCC Special Report*, 2019.  
- A. Howard et al., *Searching for MobileNetV3*, ICCV, 2019.
>>>>>>> f274292 (initial commit: MobileNetV3 multi-head plant diseade diagnostic app)

- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
=======
