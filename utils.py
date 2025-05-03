import torch
import torch.nn as nn
import torchvision.io as tv_io
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import asl
# TODO: Cleanup source
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