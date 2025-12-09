[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WG_g337P)
# E4040 2025 Fall Project
## TODO: Finetuned Neural Image Caption Generator

Repository for E4040 2025 Fall Project
  - Distributed as Github repository and shared via Github Classroom
  - Contains only `README.md` file

## Project Instructions
Please read the project instructions carefully. In particular, pay extra attention to the following sections in the project instructions:
 - [Obligatory Github project updates](https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0#bookmark=id.8ga1w2quwf7y)
 - [Student Contributions to the Project](https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0#bookmark=id.3jlnclcqaru7)

The project instructions can be found here:
https://docs.google.com/document/d/1DbWKjFzJg8_-KNG4YRsV-8WdLUgiiGfqTqjwm6-6Hcg/edit?tab=t.0 

## TODO: This repository is to be used for final project development and documentation, by a group of students
  - Students must have at least one main Jupyter Notebook, and a number of python files in a number of directories and subdirectories such as `utils` or similar, as demonstrated in the assignments
  - The content of this `README.md` should be changed to describe the actual project
  - The organization of the directories has to be meaningful

## Detailed instructions how to submit this project:
1. The project will be distributed as a Github classroom assignment - as a special repository accessed through a link
2. A student's copy of the assignment gets created automatically with a special name
3. **Students must rename the repository per the instructions below**
5. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud
6. If some model is too large to be uploaded to Github - 1) create google (liondrive) directory; 2) upload the model and grant access to e4040TAs@columbia.edu; 3) attach the link in the report and this `README.md`
7. Submit the report as a PDF in the root of this Github repository
8. Also submit the report as a PDF in Courseworks
9. All contents must be submitted to Gradescope for final grading

## TODO: (Re)naming of a project repository shared by multiple students
Students must use a 4-letter groupID, the same one that was chosen in the class spreadsheet in Google Drive: 
* Template: e4040-2025Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2025Fall-Project-MEME-zz9999-aa9999-aa0000.

# Organization of this directory
To be populated by students, as shown in previous assignments.
```

TODO: Create a directory/file tree
```
# Show-and-Tell Image Captioning (TensorFlow 1.15 GPU)

## Overview
This repo implements the classic Show-and-Tell model for automatic image captioning, based on **TensorFlow 1.15** (GPU) and Python 3.6.  
All dependencies are packed into a **single Docker image** so you can train/evaluate on any machine with an NVIDIA GPU in minutes.

---

## Prerequisites
- NVIDIA GPU driver ≥ 418
- [Docker](https://docs.docker.com/get-docker/) & [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker)

---

## Quick Start with Docker

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

- experiment 1 Frozen CNN and Training RNN 
   ```bash
   python main.py --phase=train \
               --open_atten=False \
               --load_cnn \
               --cnn_model_file=vgg16_weights.npz
   ```
- experiment 2 Finetuning: Based on the experiment 1 best model checkpoints to finetuning the whole architecture(CNN & RNN )
     ```bash
   python3 main.py --phase=train \
              --open_atten=False \
              --load \
              --model_file='./models/experiment_1/112900.npy' \
              --train_cnn \
   ```
- experiment 3 Joint training: Unfrozen CNN and Training the whole architecture (CNN & RNN)
     ```bash
      python main.py --phase=train \
               --open_atten=False \
               --load_cnn \
               --train_cnn \
               --cnn_model_file=vgg16_weights.npz
   ```
- experiment 4 Frozen CNN and Training RNN with Attention Mechanism
     ```bash
   python main.py --phase=train \
               --open_atten=True \
               --load_cnn \
               --cnn_model_file=vgg16_weights.npz
   ```
8. **Monitor with TensorBoard**  
   Open [http://localhost:6006](http://localhost:6006) on your host.

---

## Docker one-liner for veterans
```bash
docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace -p 6006:6006 tensorflow/tensorflow:1.15.0-gpu-py3 bash -c "pip install -r requirements.txt && python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\" && apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0 && python3 main.py --phase=train --load_cnn --cnn_model_file=vgg16_weights.npz"
```

---

## Evaluation / Inference
```bash
python3 main.py --phase=test --model_file=./models/ckpt-1000000
```

---

## TensorBoard
Logs are written to `./summary` inside the container (mapped to host).  
After starting training, browse:
```
http://localhost:6006
```

---

## Notes
- The image ships with CUDA 10.0 & cuDNN 7.4, compatible with TF 1.15.  
- All Python packages listed in `requirements.txt` are installed via `pip` inside the container; no host-side Python needed.  
- If you need Jupyter, add `-p 8888:8888` and start `jupyter notebook --ip 0.0.0.0 --allow-root`.

---

## Contributing
Feel free to open issues or PRs!
