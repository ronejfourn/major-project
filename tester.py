import cv2
import datetime
import math
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torchvision import transforms 

from pathlib import Path
from swiftnet import * 
from cityscapes import *
from karesnet import *
from relukan import *
from util import *
from resnet import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'swiftnet_ResNet9_512x256cityscapes_nearest_128_19_best_model.pt'
model = SwiftNet(backbone = ResNet9)
checkpoint = torch.load(model_path)
model = model.cuda().eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.286, 0.325, 0.283], std=[0.176, 0.180, 0.177])
])

palette = CityScapes.train_id_to_color 
cap = cv2.VideoCapture('challenge.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('segmented_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (512, 256))
    input_tensor = transform(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0]
        pred = torch.argmax(output, dim=0).cpu().numpy()

    color_mask = palette[pred]
    color_mask = cv2.resize(color_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

    out.write(overlay)

    cv2.imshow("Original Video", frame)
    cv2.imshow("Segmentation Output", overlay)
    cv2.moveWindow("Original Video", 100, 100)
    cv2.moveWindow("Segmented Output", 800, 100)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
