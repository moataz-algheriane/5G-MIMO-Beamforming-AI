```markdown
# 5G/6G Massive MIMO Beamforming Optimization using Cascaded Deep Learning

An end-to-end deep learning framework designed to optimize beamforming vector prediction and spatial tracking for 5G/6G Massive MIMO wireless communication systems. This project utilizes ray-tracing channel data (generated via DeepMIMO) to reduce Channel State Information (CSI) feedback overhead and replace exhaustive beam sweeps with direct neural inference.

---

## 📡 Telecom Background & System Model
In mmWave Massive MIMO systems, finding the optimal precoding vector typically requires high computational complexity and vast radio overhead. This project formulates the beam prediction task as a **Multi-Task Cascaded Learning Problem**, learning spatial channel features to simultaneously predict:
1. **Discrete Beam Index Classification:** Selecting the best beam codebook entry.
2. **Complex Value Regression:** Reconstructing channel gain/phase vectors under unit-norm power constraints.

---

## Repository Architecture
```text
├── data/       # DeepMIMO ray-tracing datasets (Hybrid & 8-Antenna setups)
├── demo/       # Interactive prediction test script (demo_run.py)
├── docs/       # Performance curves, spatial coverage maps, and evaluation reports
├── models/     # Pre-trained multi-task model (best_cascaded_model_v6_unitnorm.keras)
└── src/        # Core pipeline scripts (Data gen -> Leakage test -> Training -> Evaluation)
```

---

##  Key Technical Highlights
- **Zero Data-Leakage:** Verified train/test dataset independence (`03_data_leakage_test.py`).
- **Unit-Norm Constraint Customization:** Enforced physical power conservation in complex array weights.
- **Spatial Map Analysis:** Evaluated model generalization over non-uniform user distribution (`Hybrid_Train_Test_Spatial_Map.png`).

---

## How to Run

### Installation
Ensure you have Python 3.9+ installed, then install required dependencies:
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

### Execution Steps
1. **Data Validation:**
   ```bash
   python src/03_data_leakage_test.py
   ```
2. **Evaluate Performance Metrics:**
   ```bash
   python src/05_model_evaluation.py
   ```
3. **Run Live Demo:**
   ```bash
   python demo/demo_run.py
   ```

---

## 👨‍💻 Author
**Moataz Algheriane**  
*Telecommunications Engineer | 5G/6G Wireless Networks & MIMO Systems*  
- **LinkedIn:** [Moataz Algheriane](https://www.linkedin.com/in/moataz-algheriane-b240b834b)

```