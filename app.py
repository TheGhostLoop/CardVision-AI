import streamlit as st
from PIL import Image
import torch 
from torchvision import transforms
import torch.nn as nn 
st.set_page_config(
    page_title="CardVision AI",
    page_icon="🂡",
    layout="wide"
)


st.title("🂡 CardVision AI")

st.markdown(
"""
### Playing Card Classification using CNN

Upload an image of a playing card and let the CNN predict
its class along with prediction confidence.

---
"""
)


# model loading
@st.cache_resource
class CNN(nn.Module):
  def __init__(self,num_classes=53):
    super().__init__()
    self.structure = nn.Sequential(
        # 3 x 224 x 224
        nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),
        # 32 x 224 x 224
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 32 x 112 x 112

        nn.Conv2d(
            in_channels= 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        ),

        # 64 x 112 x 112

        nn.ReLU(),
        nn.MaxPool2d(2),
        # 64 x 56 x 56

        nn.Conv2d(
            in_channels= 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        ),
        # 128 x 56 x 56

        nn.ReLU(),
        nn.MaxPool2d(2)
        #  128 x 28 x 28
    )
    self.classifier = nn.Sequential(
        nn.Linear(128*28*28,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,num_classes)
)
    self.flatten = nn.Flatten()

  def forward(self,x):

    x = self.structure(x)
    x = self.flatten(x)
    x = self.classifier(x)

    return x

def load_model():

    model = CNN(num_classes=53)

    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    model.load_state_dict(
        torch.load("best_model_quantized.pth", map_location="cpu")
    )

    model.eval()

    return model

model = load_model()



# classes
classes  = ['ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades', 'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades', 'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades', 'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades', 'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades', 'joker', 'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades', 'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades', 'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades', 'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades', 'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades', 'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades', 'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades', 'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades']


# tranformation for the image
predict_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



uploaded_file = st.file_uploader(
    "Upload a Playing Card",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("🎯 Prediction")

        # Preprocess image
        img = predict_transform(image)
        img = img.unsqueeze(0).to("cpu")
        with st.spinner("Analyzing image..."):
            # Predict
            with torch.no_grad():
                logits = model(img)
                probabilities = torch.softmax(logits, dim=1)

                # Top prediction
                confidence, prediction = torch.max(probabilities, dim=1)

                predicted_card = classes[prediction.item()].title()
                confidence = confidence.item() * 100

                st.success(f"### 🂡 {predicted_card}")

                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2f}%"
                )

                st.progress(confidence / 100)

                if confidence >= 90:
                    st.success("High confidence prediction ✅")
                elif confidence >= 70:
                    st.info("Moderate confidence prediction")
                else:
                    st.warning("Low confidence. Try a clearer image.")

                st.markdown("---")

                st.subheader("Top 5 Predictions")

                top5_prob, top5_idx = torch.topk(probabilities, k=5)

                for prob, idx in zip(top5_prob[0], top5_idx[0]):

                    class_name = classes[idx.item()].title()
                    prob = prob.item() * 100

                    st.write(f"**{class_name}**")

                    st.progress(prob / 100)

                    st.caption(f"{prob:.2f}%")

st.markdown("---")

st.subheader("Model Information")

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.info("🧠 CNN from Scratch")

with col2:
    st.info("🃏 53 Classes")

with col3:
    st.info("⚡ Quantized (49 MB)")

with col4:
    st.info("🔥 PyTorch")

with st.sidebar:

    st.title("About")

    st.write(
        """
        **CardVision AI** is a CNN built from scratch
        using PyTorch to classify playing cards.

        ---
        """
    )

    st.markdown("### Architecture")

    st.write("""
• Conv2D (3→32)\n
• Conv2D (32→64)\n
• Conv2D (64→128)\n
• MaxPool
• ReLU
• Fully Connected
• Dynamic Quantization\n
Frameworks
• PyTorch
• Streamlit
• TorchVision

Author

Prince
    """)

    st.markdown("---")

    st.markdown("Created by **Prince**")
