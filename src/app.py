import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from lenet_digit_recognition import LeNetClassifier1
from lenet_leaf_disease import LeNetClassifier2
from textCNN import TextCNN
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from task3_preprocess import preprocess_text

# ----------------------------------------
# 🔧 Streamlit Config
# ----------------------------------------
st.set_page_config(page_title="CNN Applications", layout="wide")

# ----------------------------------------
# 🔄 Loaders
# ----------------------------------------
@st.cache_resource
def load_vocab(vocab_path):
    return torch.load(vocab_path, map_location=torch.device('cpu'))

@st.cache_resource
def load_model(model_path, num_classes, architecture):
    if architecture == 'LeNet1':
        model = LeNetClassifier1(num_classes=num_classes)
    elif architecture == 'LeNet2':
        model = LeNetClassifier2(num_classes=num_classes)
    elif architecture == 'TextCNN':
        model = TextCNN(
            vocab_size=10000,
            embedding_dim=100,
            kernel_sizes=[3, 4, 5],
            num_filters=100,
            num_classes=num_classes
        )
    else:
        raise ValueError("Unsupported architecture.")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ----------------------------------------
# 🔢 Digit Recognition
# ----------------------------------------
def inference_digit(image, model):
    if image.size[0] != image.size[1]:
        image = transforms.CenterCrop(min(image.size))(image)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)
    probs = F.softmax(predictions, dim=1)
    p_max, yhat = torch.max(probs, 1)
    return p_max.item() * 100, yhat.item()

def run_digit_recognition():
    st.title("Digit Recognition")
    st.subheader("Model: LeNet | Dataset: MNIST")
    model = load_model("model/model1.pt", num_classes=10, architecture='LeNet1')

    option = st.selectbox("Input Option", ("Upload Image File", "Run Example Image"))
    if option == "Upload Image File":
        file = st.file_uploader("Upload a digit image", type=["jpg", "png"])
        if file:
            image = Image.open(file)
            p, label = inference_digit(image, model)
            st.image(image)
            st.success(f"Digit: **{label}** | Confidence: **{p:.2f}%**")
    else:
        image = Image.open("data/Demo/demo_8.png")
        p, label = inference_digit(image, model)
        st.image(image)
        st.success(f"Example Digit: **{label}** | Confidence: **{p:.2f}%**")

# ----------------------------------------
# 🌿 Cassava Leaf Disease Classification
# ----------------------------------------
def cassava_inference(image, model):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)
    probs = F.softmax(predictions, dim=1)
    p_max, yhat = torch.max(probs, 1)
    return p_max.item() * 100, yhat.item()

def run_cassava_leaf_disease():
    st.title("Cassava Leaf Disease Classification")
    st.subheader("Model: LeNet | Dataset: Cassava Leaf")
    model = load_model("model/model2.pt", num_classes=5, architecture='LeNet2')

    cassava_classes = {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy"
    }

    option = st.selectbox("Input Option", ("Upload Image File", "Run Example Image"))
    if option == "Upload Image File":
        file = st.file_uploader("Upload a cassava leaf image", type=["jpg", "png"])
        if file:
            image = Image.open(file)
            p, label = cassava_inference(image, model)
            st.image(image)
            st.success(f"Predicted: **{cassava_classes[label]}** | Confidence: **{p:.2f}%**")
    else:
        image = Image.open("data/Demo/demo_cbb.jpg")
        p, label = cassava_inference(image, model)
        st.image(image)
        st.success(f"Predicted: **{cassava_classes[label]}** | Confidence: **{p:.2f}%**")


def main():
    st.sidebar.title("🧠 Choose a Task")
    task = st.sidebar.radio("Select Task", ["Digit Recognition", "Cassava Leaf Disease", "Other Tasks"])
    
    if task == "Digit Recognition":
        run_digit_recognition()
    elif task == "Cassava Leaf Disease":
        run_cassava_leaf_disease()
    elif task == "Other Tasks":
        #comming soon
        st.write("Other tasks are coming soon!")

if __name__ == "__main__":
    main()
