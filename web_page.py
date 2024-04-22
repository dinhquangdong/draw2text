import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

import streamlit as st
from streamlit_drawable_canvas import st_canvas

import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


@st.cache_resource
def load_model(device: str) -> None:
    model = Net()
    model.load_state_dict(torch.load('./model/results/model.pth'))
    model.to(device)
    model.eval()
    return model


## INSTANCES
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    model = load_model(DEVICE)
    
    # uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])
    # if uploaded_file is not None:
    #     bytes_data = uploaded_file.getvalue()
    #     image = Image.open(BytesIO(bytes_data))
        
    #     tensor = transforms.Compose([
    #         transforms.Resize((28, 28)),
    #         transforms.ToTensor()
    #     ])(image).to(DEVICE)
        
    #     st.write(tensor.shape)
        
    #     pred = model(tensor)
    #     output = pred.argmax(dim=1).item()

    #     st.image(bytes_data, caption=f'Dự đoán: {output}', width=300)
    
    
    st.subheader('Canvas')
    # Specify canvas parameters in application
    drawing_mode = 'freedraw'

    stroke_width = st.sidebar.slider("Độ dày nét vẽ: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Chọn màu cho nét vẽ: ", "#000")
    bg_color = st.sidebar.color_picker("Chọn màu nền: ", "#fff")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color='rgba(255, 165, 0, 0.3)',  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        key='canvas',
    )
    
    # Khi ấn nút bấm Dự đoán
    if st.button('Dự đoán'):
        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            
            # Chuyển đổi ảnh sang không gian màu xám
            gray_image = Image.fromarray(img).convert('L')
            gray_array = np.array(gray_image)
            gray_array[gray_array == 255] = 0
            gray_array[gray_array != 0] = 255
            
            
            resized_image = Image.fromarray(gray_array).resize((28, 28))
            st.image(resized_image)
            
            # Chuyển đổi ảnh thành tensor PyTorch
            tensor = torch.tensor(np.array(resized_image), dtype=torch.float32).unsqueeze(0)
            tensor = tensor.to(DEVICE)
            
            output = model(tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
            st.write(pred)
            
