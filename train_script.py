import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
img_dir = "/scratch/bojan/large_sample_jpg/"
file_writer = "/scratch/bojan/TFCNN/160"
model_path = "/scratch/bojan/model_verso.ckpt"

from train_eval import run_training
from utils import loading_training_data
import numpy as np

images, labels = loading_training_data(img_dir, recto=False, size=400)
labels = np.argmax(labels, axis=3)
run_training(images, labels, filter_size=11, skip=3, learning_rate=1e-5, batch_size=1, epochs=20, num_clases=2, gpu_memory_fraction=0.7, file_writer=file_writer, model_path=model_path)
print('Done with training')
