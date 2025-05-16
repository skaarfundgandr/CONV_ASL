import torch
import torch.nn as nn
import torchvision.io as tv_io
import numpy as np
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import mediapipe as mp
from PIL import Image

import asl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNN(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()

    return correct / N

def train(model, train_loader, train_N, random_trans, optimizer, loss_function):
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(random_trans(x))
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print(f"Train - Loss: {loss:.4f} Accuracy: {accuracy:.4f}")

def train_model():
    return asl.train_model()

def validate(model, valid_loader, valid_N, loss_function):
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print(f"Valid - Loss: {loss:.4f} Accuracy: {accuracy:.4f}")

def import_model(model_path):
    return torch.load(model_path, map_location=device, weights_only=False)

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')

def predict_letter(model, file_path):
    IMG_WIDTH = 28
    IMG_HEIGHT = 28

    alphabet = "abcdefghiklmnopqrstuvwxy"

    preprocess_trans = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.Grayscale()  # From Color to Gray
    ])

    # Load and grayscale image
    image = tv_io.decode_image(file_path, tv_io.ImageReadMode.GRAY)
    # Transform image
    image = preprocess_trans(image)
    # Batch image
    image = image.unsqueeze(0)
    # Send image to correct device
    image = image.to(device)
    # Make prediction
    output = model(image)
    # Find max index
    prediction = output.argmax(dim=1).item()
    # Convert prediction to letter
    predicted_letter = alphabet[prediction]
    # Return prediction
    return predicted_letter

def predict_from_image(model, image):
    IMG_WIDTH = 28
    IMG_HEIGHT = 28

    alphabet = "abcdefghiklmnopqrstuvwxy"

    preprocess_trans = transforms.Compose([
        transforms.ToImage(), # Converts PIL image to tensor
        transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.Grayscale()  # From Color to Gray
    ])
    try:
        cropped_image = crop_to_hand(image)
        image_tensor = preprocess_trans(cropped_image)
    except ValueError as v:
        print(v)
        image_tensor = preprocess_trans(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    # Make prediction
    output = model(image_tensor)
    # Find max index
    prediction = output.argmax(dim=1).item()
    # Convert prediction to letter
    predicted_letter = alphabet[prediction]
    # Return prediction
    return predicted_letter

def crop_to_hand(image, padding = 60):
    mp_hands = mp.solutions.hands
    
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    source_img = cv_image.copy()
    height, width = cv_image.shape[:2]
    
    with mp_hands.Hands(static_image_mode=True,
                  max_num_hands=1,
                  min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            raise ValueError("No hand detected!")
        
        landmark = results.multi_hand_landmarks[0].landmark
        xs = [int(pt.x * width) for pt in landmark]
        ys = [int(pt.y * height) for pt in landmark]
    
    x0 = max(min(xs) - padding, 0)
    y0 = max(min(ys) - padding, 0)
    x1 = min(max(xs) + padding, width)
    y1 = min(max(ys) + padding, height)
    cropped_cv = source_img[y0:y1, x0:x1]
    
    cropped_rgb = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB)
    
    cropped_img = Image.fromarray(cropped_rgb)
    
    return cropped_img