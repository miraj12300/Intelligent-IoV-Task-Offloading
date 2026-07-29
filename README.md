# 🚗 Intelligent IoV Task Offloading using DRL and ACO

> A research project that explores intelligent task offloading in the Internet of Vehicles (IoV) by comparing **Deep Reinforcement Learning (DVTP)** and **Ant Colony Optimization (ACO)** for mobility-aware edge computing.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

---

## 📖 Overview

The increasing computational demands of connected and autonomous vehicles require efficient task scheduling strategies. This project investigates two intelligent approaches for task offloading:

- **Deep Reinforcement Learning (DVTP Framework)**
- **Ant Colony Optimization (ACO)**

The objective is to optimize task execution by reducing latency, improving resource utilization, balancing computational load, and minimizing communication overhead in a mobility-aware edge-cloud environment.

---

## ✨ Features

- 🚘 Intelligent task offloading for IoV
- 🧠 Transformer-based trajectory prediction
- 📊 Graph Attention Network (GAT) for task graph representation
- 🎯 PPO-based reinforcement learning policy
- 🐜 Ant Colony Optimization (ACO) scheduler
- 📈 Performance visualization and analysis
- 🔥 Heatmaps and convergence analysis
- 📉 Resource utilization evaluation
- 📑 Research paper included

---

# 📂 Repository Structure

```text
Intelligent-IoV-Task-Offloading/
│
├── README.md
├── LICENSE
│
├── paper/
│   └── Intelligent_IoV_Task_Offloading.pdf
│
├── images/
│   └── (architecture diagrams, workflow, outputs)
│
├── results/
│   └── (generated graphs and evaluation results)
│
└── src/
    ├── dvtp_framework.py
    ├── aco_optimization.py
    ├── aco_visualization.py
    ├── evaluation.py
    ├── requirements.txt
    └── synthetic_vehicle_trajectory_data.csv
```

---

# 🏗️ Modules

## 🧠 DVTP Framework

Implements a Deep Reinforcement Learning based Dynamic Vehicle Task Planning framework.

### Components

- Transformer-based trajectory prediction
- Graph Attention Network (GAT)
- PPO reinforcement learning agent
- DAG-based task scheduling environment

---

## 🐜 ACO Optimization

Implements an Ant Colony Optimization algorithm for intelligent task scheduling.

Features include:

- Pheromone update mechanism
- Heuristic evaluation
- Task-to-node assignment
- Makespan optimization
- Resource-aware scheduling

---

## 📊 Visualization Module

Generates performance analysis for ACO results, including:

- Pheromone heatmaps
- Heuristic heatmaps
- Probability heatmaps
- Confusion matrix
- Resource utilization analysis
- Convergence analysis
- Accuracy reports
- Performance dashboard

---

## 📈 Evaluation Module

Provides performance evaluation and comparison of scheduling algorithms using metrics such as:

- Makespan
- Resource Utilization
- Load Balancing
- Scheduling Accuracy
- Communication Cost
- Energy Efficiency
- Convergence Performance

---

# 📊 Performance Metrics

The project evaluates the following metrics:

- Makespan
- Latency
- Throughput
- Resource Utilization
- Load Balancing
- Communication Cost
- Energy Consumption
- Scheduling Accuracy

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- PyTorch
- PyTorch Geometric
- NetworkX
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

# 📄 Research Paper

The repository includes the research paper describing:

- Problem Statement
- Literature Review
- Proposed Methodology
- System Architecture
- Experimental Analysis
- Performance Evaluation
- Conclusion

---

# 🚀 Future Work

- Complete integration of Transformer and GAT with PPO training
- Real-world IoV datasets
- Multi-edge server deployment
- Cloud-edge collaborative scheduling
- Additional optimization algorithms
- Real-time simulation platform
- Comparative benchmarking with existing methods

---

# ▶️ Getting Started

### Clone the repository

```bash
git clone https://github.com/<your-username>/Intelligent-IoV-Task-Offloading.git
```

### Install dependencies

```bash
pip install -r src/requirements.txt
```

### Run the project

```bash
python src/dvtp_framework.py
```

Run the visualization module:

```bash
python src/aco_visualization.py
```

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**SK Miraj**

B.Tech – Computer Science & Engineering

Research Interests

- Internet of Vehicles (IoV)
- Artificial Intelligence
- Deep Reinforcement Learning
- Edge Computing
- Optimization Algorithms
- Intelligent Transportation Systems

---

## ⭐ Support

If you found this repository useful, consider giving it a **⭐ Star**.
