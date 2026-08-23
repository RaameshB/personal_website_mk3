---
{"dg-publish":true,"permalink":"/resume/","updated":"2026-08-23T10:12:14.192-05:00","dg-note-properties":{}}
---

# Resume
Conroe, TX • (936) 828-7896 • raamesh@raameshb.xyz • [GitHub](https://github.com/RaameshB)

## Selected Research Projects

### Mamba-1 in JAX — Independent Research
*June 2026 – Present*

[github.com/RaameshB/JAX-Mambas](https://github.com/RaameshB/JAX-Mambas)

- Implemented a configurable Mamba-1 model in JAX and Flax NNX with configurations faithful to both the paper and the official repository; supports exponential Euler and zero-order hold (ZOH) discretization and both broadcasted scalar and learned generalized low-rank $\Delta$ selection.
- Integrated Mamba’s official CUDA selective scan kernel into JAX via XLA FFI; validated CUDA forward and reverse mode selective scan agreement for exponential Euler and CUDA forward agreement for ZOH, whose reverse mode uses a JAX fallback.
- Reproduced Mamba’s induction heads length generalization with 100% accuracy at every tested length through 32,768—the longest evaluated due to compute constraints—after training only at length 256 (128 evaluation samples per length).
- On an A100, achieved 1.42× whole-model loss and gradient throughput (D = 64) and up to 1.84× CUDA selective-scan forward throughput at L = 8,192 (B = 8, D = 64, N = 16) versus `jax.lax.associative_scan` baseline with `jax.remat`.

### Structured State Space Models — S4 and S5 Implementations
*2026*

- Implemented S4 in JAX using bilinear discretization and FFT-based convolution for its parallel form, following the Annotated S4.

  [github.com/RaameshB/Replicating-S4](https://github.com/RaameshB/Replicating-S4)

- Implemented S5 in Equinox using JAX’s parallel associative scan primitive.

### Orbit Wars Kaggle Competition — Reinforcement Learning Agent
*June 2026 – July 2026*

- Built a JAX reinforcement learning pipeline combining behavioral-cloning pretraining with PPO and SVG.
- Reimplemented the competition environment as a differentiable JAX simulator to support gradient-based optimization through environment dynamics; trained and submitted an agent.

### Common Crawl Topic Modeling — DS 410 Project
*Oct. 2025 – Dec. 2025*

[github.com/RaameshB/ds410-course-proj-redo](https://github.com/RaameshB/ds410-course-proj-redo)

- Built an end-to-end PySpark pipeline for distributed ingestion, preprocessing, and topic modeling over multi-terabyte Common Crawl web datasets.

## Open-Source Contributions

### EchoTorch — Contributor
*May 2024 – Nov. 2024*

[github.com/nschaetti/EchoTorch](https://github.com/nschaetti/EchoTorch)

- Modernized a PyTorch Echo State Network library for Python 3, updating dependencies and CI while repairing training, parameter-registration, and reproducibility issues across 19 commits merged upstream in two pull requests.

## Experience

### OSSIG Research Lab, Penn State — Lead Student Researcher
*Sept. 2024 – May 2025*

- Designed an asynchronous multi-node hyperparameter-search system in Python: independent CMA-ES workers trained and evaluated Echo State Networks while coordinating trials through a centralized PostgreSQL database; led the lab’s machine learning research and mentored student contributors.

### CavBots, FRC 7492 — Computer Vision Head
*Dec. 2023 – May 2024*

- Deployed YOLOv8 and RT-DETR through TensorRT on an NVIDIA Jetson Orin Nano and built a real-time pipeline to detect objects and pinpoint their positions in 3D space using Intel RealSense point clouds, RANSAC floor removal, and geometric object-center estimation.

### Aapoon — Machine Learning Intern
*May 2023–Aug. 2023*

- Prototyped liveness-detection pipelines in Python with OpenCV and MediaPipe for integration into a facial-authentication system.

### Phantom, FTC 12857 — Robotics Team Captain
*Aug. 2022 – May 2023*

- Led control and software development during a team rebuild; prototyped OpenCV vision pipelines in Python, deployed robot code in Java, and created technical training materials for new members.

## Technical Skills

**Languages:** Python, Java, SQL, Bash

**Machine Learning:** JAX ML stack (Flax NNX, Equinox, Optax, Orbax), PyTorch, Hugging Face Transformers, scikit-learn

**Models & Methods:** Transformers, State Space Models, Mamba, S4, PPO, behavioral cloning, CMA-ES, distributed training, model evaluation

**Systems & Data:** Linux, Docker, Spark/PySpark, TensorRT, AWS EC2/S3, PostgreSQL, Pandas, Polars

## Education

### Pennsylvania State University, University Park
*Expected December 2027*

**B.S. Computational Data Science**  
GPA: 3.93 • Dean’s List

**Relevant Coursework**
- MATH 452 — Mathematics of Deep Learning Algorithms
- STAT 414 — Probability
- MATH 220 — Linear Algebra
- DS 410 — Programming for Big Data (distributed computing, Spark/PySpark)