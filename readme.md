# Show-and-Tell Image Captioning (TensorFlow 1.15 GPU)

## Overview
This repo implements the classic Show-and-Tell model for automatic image captioning, based on **TensorFlow 1.15** (GPU) and Python 3.6.  
All dependencies are packed into a **single Docker image** so you can train/evaluate on any machine with an NVIDIA GPU in minutes.

---

## 🔧 Prerequisites
- NVIDIA GPU driver ≥ 418
- [Docker](https://docs.docker.com/get-docker/) & [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker)

---

## ⚡ Quick Start with Docker

1. **Pull the official TF 1.15 GPU image**
   ```bash
   docker pull tensorflow/tensorflow:1.15.0-gpu-py3
   ```

2. **Clone / enter the repo**
   ```bash
   git clone <your-repo-url> show_and_tell && cd show_and_tell
   ```

3. **Launch the container**
   ```bash
   docker run --gpus all -it --rm \
     -v $(pwd):/workspace \
     -w /workspace \
     -p 6006:6006 \
     tensorflow/tensorflow:1.15.0-gpu-py3 \
     bash
   ```

4. **Inside container - install deps & data**
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   
   # system libs for OpenCV
   apt-get update && apt-get install -y \
     libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0
   ```

5. **Clean old weights (optional)**
   ```bash
   rm -rf ./train/*.npy ./models/*
   ```

6. **Train**
   ```bash
   python3 main.py --phase=train --load_cnn --cnn_model_file=vgg16_weights.npz
   ```

7. **Monitor with TensorBoard**  
   Open [http://localhost:6006](http://localhost:6006) on your host.

---

## 📂 Docker one-liner for veterans
```bash
docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace -p 6006:6006 tensorflow/tensorflow:1.15.0-gpu-py3 bash -c "pip install -r requirements.txt && python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\" && apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0 && python3 main.py --phase=train --load_cnn --cnn_model_file=vgg16_weights.npz"
```

---

## 🧪 Evaluation / Inference
```bash
python3 main.py --phase=test --model_file=./models/ckpt-1000000
```

---

## 📈 TensorBoard
Logs are written to `./summary` inside the container (mapped to host).  
After starting training, browse:
```
http://localhost:6006
```

---

## 📝 Notes
- The image ships with CUDA 10.0 & cuDNN 7.4, compatible with TF 1.15.  
- All Python packages listed in `requirements.txt` are installed via `pip` inside the container; no host-side Python needed.  
- If you need Jupyter, add `-p 8888:8888` and start `jupyter notebook --ip 0.0.0.0 --allow-root`.

---

## 🤝 Contributing
Feel free to open issues or PRs!
```
