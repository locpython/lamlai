import base64
from tensorflow.keras.models import load_model
import requests
import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image
import tensorflow as tf
print(tf.__version__)
def main():
    st.title("Project: Recognized handwritting")
    model = load_model('my_cnn_model.h5')

    def preprocess_image(image):
        # Convert PIL Image to NumPy array
        image_np = np.array(image)
        
        # Resize image using OpenCV's Lanczos interpolation
        img_resized = cv2.resize(image_np, (64, 64), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to float, add batch dimension, and normalize
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        return img_array

    def load_image_from_base64(base64_str):
        base64_str = base64_str.split(",")[1]  # Remove the data URL part
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))

    st.title('Nhận diện hình ảnh từ URL hoặc Base64')

    # Input for image URL or Base64 string
    image_input = st.text_area("Nhập URL hình ảnh hoặc chuỗi base64")

    if st.button("ĐOÁN MÒ"):
        if image_input:
            try:
                if image_input.startswith('data:image/'):
                    # Handle Base64 image
                    image = load_image_from_base64(image_input)
                else:
                    # Handle URL image
                    response = requests.get(image_input)
                    image = Image.open(io.BytesIO(response.content))
                
                # Display the image
                st.image(image, caption='Hình ảnh đã tải lên', use_column_width=True)
                
                # Preprocess image and predict
                image_array = preprocess_image(image)
                predictions = model.predict(image_array)
                
                # Interpret prediction result
                if predictions[0][0] >= 0.8:
                    confidence = predictions[0][0]
                    st.write(f'CON CAOAO')
                    st.write(f'Doanh số dự đoán: {confidence * 100:.2f}%')
                else:
                    confidence = 1 - predictions[0][0]
                    st.write(f'MỒM LÈO')
                    st.write(f'Doanh số dự đoán: {confidence * 100:.2f}%')
            
            except Exception as e:
                st.error(f"Không thể xử lý hình ảnh: {e}")
        else:
            st.write('Xin nhập URL hoặc chuỗi base64 hình ảnh')

            

       
if __name__ == "__main__":
    main()