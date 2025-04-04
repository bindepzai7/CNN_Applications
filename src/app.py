import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
st.set_page_config(page_title="CNN Applications", layout="wide")

from PIL import Image
import torchvision.transforms as transforms
from lenet_digit_recognition import LeNetClassifier1

@st.cache_resource
def load_model(model_path, num_classes, architecture):
    if architecture == 'LeNet1':
        model = LeNetClassifier1(num_classes=num_classes)
    else:
        raise ValueError("Unsupported architecture. Only 'LeNet' is supported.")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model    

model = load_model('model/model1.pt', num_classes=10, architecture='LeNet1')

def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size
    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_new = img_transform(image)
    img_new = img_new.expand(1, 1, 28, 28)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item()*100, yhat.item()

def run_digit_recognition():
    st.title('Digit Recognition')
    st.subheader('Model: LeNet | Dataset: MNIST')

    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))

    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image of a digit", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The uploaded image is of the digit {label} with {p:.2f} % probability.") 

    elif option == "Run Example Image":
        image = Image.open('data/Demo/demo_8.png')
        p, label = inference(image, model)
        st.image(image)
        st.success(f"The image is of the digit {label} with {p:.2f} % probability.")
        
def run_cassava_leaf_disease():
    st.title('Cassava Leaf Disease Classification')
    st.subheader('Model: LeNet | Dataset: Cassava Leaf Disease')

    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))

    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image of a cassava leaf", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The uploaded image is of the cassava leaf with {p:.2f} % probability.") 

    elif option == "Run Example Image":
        image = Image.open('data/Demo/demo_cbb.jpg')
        p, label = inference(image, model)
        st.image(image)
        st.success(f"The image is of the cassava leaf with {p:.2f} % probability.")


def main():

    st.sidebar.title("Choose a Task")
    task = st.sidebar.radio("Select a task:", ["Digit Recognition", "Cassava Leaf Disease", "Sentiment Analysis", "Other Task"])
    
    if task == "Digit Recognition":
        run_digit_recognition()
    elif task == "Other Task":
        st.info("Coming soon!")

if __name__ == '__main__':
    main() 