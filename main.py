
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import io
from torchvision import transforms

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TorchScript model
model = torch.jit.load("/kaggle/input/final_model/pytorch/default/1/model_final.pt")
device = torch.device("cpu")  # Force to CPU
model.to(device)
model.eval()

# Define the image preprocessing function
def get_transforms():
    return transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor(),
    ])

def preprocess_and_predict(model, img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Compute mean and std
    mean = torch.mean(img_tensor)
    std = torch.std(img_tensor)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor, mean.unsqueeze(0).to(device), std.unsqueeze(0).to(device))
        predicted_class = output.argmax(dim=1).item()

    return predicted_class, mean.item(), std.item()

# Define the response model
class PredictionResponse(BaseModel):
    predicted_class: int
    mean: float
    std: float
    description: str

@app.post("/predict/", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Preprocess the image
    transform = get_transforms()
    img_tensor = transform(image)

    # Run the prediction
    predicted_class, mean, std = preprocess_and_predict(model, img_tensor)

    # Map predicted class to dementia type
    dementia_types = {0: "Mild Dementia", 1: "Moderate Dementia", 2: "Non Demented", 3: "Very Mild Dementia"}
    description = dementia_types.get(predicted_class, "Not Supported")

    return PredictionResponse(
        predicted_class=predicted_class,
        mean=mean,
        std=std,
        description=description
    )

# Add a root route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Dementia Prediction API. Use the /predict/ endpoint to upload images."} 