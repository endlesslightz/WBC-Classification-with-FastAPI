import io

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms

# Initialize FastAPI app
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,  # how big is the square that's going over the image?
                stride=1,  # default
                padding=1,
            ),  # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


class MyCustomModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(MyCustomModel, self).__init__()
        self.model = models.resnet50(weights="DEFAULT")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True), nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


# Load model weights
model = MyCustomModel(num_classes=5)
model.load_state_dict(torch.load("best.pt", map_location=device))
model = model.to(device)
model.eval()

# Define image transformation pipeline
transform = transforms.Compose(
    [
        # Resize the images to 224x224
        transforms.Resize(size=(224, 224)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    ]
)

# Define class names (from dataset)
class_names = [
    'Basofil', 
    'Eosinofil', 
    'Limfosit', 
    'Monosit', 
    'Neutrofil'
]

# Helper function to preprocess image
def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)


# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image from upload
        image = Image.open(io.BytesIO(await file.read()))

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            test_pred_labels = outputs.argmax(dim=1)
            predicted_class = class_names[test_pred_labels.item()]

        # Return prediction as JSON response
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
