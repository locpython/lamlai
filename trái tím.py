# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)

# link_train = r"C:\Users\lusan\OneDrive\Desktop\Machine learning\udemy\Machine Learning A-Z (Codes and Datasets)\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\Python\dataset\training_set"

# training_set = train_datagen.flow_from_directory(link_train,
#                                                  target_size = (64, 64),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')

# test_datagen = ImageDataGenerator(rescale = 1./255)

# link_test = r'C:\Users\lusan\OneDrive\Desktop\Machine learning\udemy\Machine Learning A-Z (Codes and Datasets)\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\Python\dataset\test_set'

# test_set = test_datagen.flow_from_directory(link_test,
#                                             target_size = (64, 64),
#                                             batch_size = 32,
#                                             class_mode = 'binary')

# cnn = tf.keras.models.Sequential()

# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# cnn.add(tf.keras.layers.Flatten())

# cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

model_save_path = r'C:\Users\lusan\OneDrive\Desktop\Machine learning\udemy\Machine Learning A-Z (Codes and Datasets)\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\Python\my_cnn_model.h5'

# cnn.save(model_save_path)

from tensorflow.keras.models import load_model
model = load_model(model_save_path)
import numpy as np
from keras.preprocessing import image
link_image = r"C:\Users\lusan\OneDrive\Desktop\Machine learning\udemy\Machine Learning A-Z (Codes and Datasets)\Part 8 - Deep Learning\Section 40 - Convolutional Neural Networks (CNN)\Python\download.jpg"
test_image = image.load_img(link_image, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
# training_set.class_indices

if result[0][0] >= 0.8:
  prediction = 'dog'
else:
  prediction = 'cat'
  
print(prediction)
print(result)


import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import bz2
import time
import io
impo
# Hàm để tải và xử lý dữ liệu
def load_and_preprocess_data():
    with bz2.open('dataset_X.pkl.bz2', 'rb') as file:
        X = pickle.load(file)
    with bz2.open('dataset_y.pkl.bz2', 'rb') as file:
        y = pickle.load(file)
    return X, y

# Hàm để xây dựng và huấn luyện mô hình
def build_and_train_model(X_train, y_train, num_classes, epochs=10, test_size=0.2):
    label_encoder = LabelEncoder()
    y_train_encoded = to_categorical(label_encoder.fit_transform(y_train))
    
    model = Sequential([
        Input(shape=X_train.shape[1:]),
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X_train, y_train_encoded, test_size=test_size, random_state=42)
    history = model.fit(X_train, y_train_encoded, epochs=epochs, validation_split=0.1, verbose=1)
    evaluation_result = model.evaluate(X_test, y_test_encoded)
    
    return model, history, evaluation_result
# Hàm để hiển thị lịch sử huấn luyện
def display_training_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(history.history['loss'], label='Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    
    axs[1].plot(history.history['accuracy'], label='Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epochs')
    
    st.pyplot(fig)

# Hàm để hiển thị mẫu dữ liệu
def show_dataset_examples(X, y):
    classes = np.unique(y)
    fig, axs = plt.subplots(len(classes), 8, figsize=(10, 2 * len(classes)))
    for i, class_label in enumerate(classes):
        indices = np.where(y == class_label)[0]
        for j in range(8):
            if len(indices) > j:
                img = X[indices[j]]
                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].axis('off')
    st.pyplot(fig)

# Hàm để dự đoán ảnh từ bảng vẽ
def predict_img(img, model):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    resized_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LANCZOS4)
    resized_img = resized_img / 255.0
    prediction = model.predict(np.expand_dims(resized_img, axis=0))
    top_3_indices = np.argsort(prediction[0])[::-1][:3]
    letters = [chr(i) for i in range(65, 91)]
    predictions = []
    for index in top_3_indices:
        predicted_letter = letters[index]
        confidence = prediction[0][index] * 100
        predictions.append(f"Character: {predicted_letter}, Confidence: {confidence:.2f}%")
    return predictions

# Hàm để vẽ và dự đoán
def draw_test_confident(model):
    col1, col2 = st.columns([2, 1])  
    with col1:
        canvas_placeholder = st.empty()

    # Place file uploader in the second column
    with col2:
        uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=['jpg', 'jpeg', 'png'], key='file_uploader')

    # Now let's add the canvas to the placeholder in the first column
    with canvas_placeholder.container():
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fill color changes for better visibility
            stroke_color="white",  # Stroke color for drawing
            background_color="black",  # Canvas background
            update_streamlit=True,
            width=200,
            height=200,
            drawing_mode="freedraw",
            stroke_width=20,
            key="canvas",
        )
    if col1.button('Predict Drawing'):
        if canvas_result.image_data is not None:
            prediction = predict_img(canvas_result.image_data.astype(np.uint8),model)
            for line in prediction:
                st.write(line)
        st.success('Thành Công', icon="✅")
        st.snow()
        st.balloons()
                
    if col2.button('Predict Image') and uploaded_file is not None:
    # Open and convert the uploaded image to numpy array
        image = Image.open(uploaded_file)
        image_2 = Image.open(uploaded_file).copy()
        
        image_array = np.array(image)
        prediction = predict_img(image_array, model)

        # Display prediction results
        for line in prediction:
            col2.write(line)
            
        # Display the uploaded image
        resized_image = cv2.resize(image_array, (200, 200), interpolation=cv2.INTER_LANCZOS4)
        image_display = Image.fromarray(resized_image)
        col1.image(image_display, caption='Uploaded Image')


# Hình 2
        img_gray = np.array(image_2.convert('L'))
        img_gray_resized = cv2.resize(img_gray, (200, 200), interpolation=cv2.INTER_LANCZOS4)
        img_gray_display = Image.fromarray(img_gray_resized, 'L')  # Specify mode as 'L' (grayscale)
        col1.image(img_gray_display, caption='Gray Image')

        # Predict using the grayscale image converted back to three channels
        prediction_2 = predict_img(img_gray_display, model)
        st.success('Thành Công', icon="✅")
        st.snow()
        st.balloons()
        
        # Add empty lines
        for _ in range(8):
            col2.write("")

        for line in prediction_2:
            col2.write(line)
# Hàm chính để chạy ứng dụng
def main():
    st.title("Project: Recognized handwritting")
    menu = ["Huấn luyện model","Dataset", "Thử nghiệm model", "chạy mô hình chó mèo"]
    choice = st.sidebar.selectbox("Chọn trang", menu)

    if 'data_loaded' not in st.session_state:
        st.session_state.X, st.session_state.y = load_and_preprocess_data()
        st.session_state.data_loaded = True

    if choice == "Huấn luyện model":
        st.header("Huấn luyện mô hình nhận dạng ký tự")
        st.write('Khoảng 500 hình / chữ cái')
        epochs = st.number_input('Chọn số epochs:', min_value=1, max_value=100, value=10)
        test_size = st.slider('Tỉ lệ dữ liệu kiểm tra:', min_value=0.1, max_value=0.9, value=0.2)
                                                         
        if st.button("Huấn luyện mô hình"):
            if 'model_trained' not in st.session_state:
                with st.spinner('Mô hình đang được huấn luyện... Vui lòng đợi.'):
                    start_time = time.time()
                    st.session_state.model, st.session_state.history, st.session_state.eval_result = build_and_train_model(st.session_state.X, 
                                                                                        st.session_state.y, 
                                                                                        num_classes=np.unique(st.session_state.y).shape[0], 
                                                                                        epochs=epochs, test_size=test_size)
                    st.session_state.model_trained = True
                    end_time = time.time() 
                    st.session_state.training_time = end_time - start_time      
        if 'model_trained' in st.session_state:
            st.write(f"Độ chính xác trên tập kiểm tra: {st.session_state.eval_result[1] * 100:.2f}%")
            st.write(f"Thời gian huấn luyện mô hình: {st.session_state.training_time:.2f} giây.") 
            display_training_history(st.session_state.history)
            draw_test_confident(st.session_state.model)
    elif choice == "Dataset":
        st.header("Hiển thị mẫu từ Dataset")
        show_dataset_examples(st.session_state.X, st.session_state.y)
        
        
    elif choice == "Thử nghiệm model":
        st.header("Thử nghiệm model")
        st.write("Đây là model mình làm với lượng dataset lớn khoảng 130000 hình cùng với 20 epochs và áp dụng 1 số kĩ thuật huấn luyện mô hình do  mình tham khảo")

        if 'loaded_model' not in st.session_state:
            st.session_state.loaded_model = load_model('model.keras')
        
        model = st.session_state.loaded_model
        label_encoder = LabelEncoder()
        if 'X' in st.session_state and 'y' in st.session_state:
            X, y = st.session_state.X, to_categorical(label_encoder.fit_transform(st.session_state.y))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            eval_result = model.evaluate(X_test, y_test)
            st.write(f"Độ chính xác trên tập kiểm tra: {eval_result[1] * 100:.2f}% (độ chính xác cao hơn khoảng 20%)")
            st.write(f"=> ta nhận thấy rằng việc epochs gấp đôi cùng với data khoảng 135000 hình so với epochs 10 và khoảng 13000 hình có sự phân biệt lớn")
            draw_test_confident(model)

    elif choice == "chạy mô hình chó mèo":
        import base64
        model = tf.keras.models.load_model('my_cnn_model.h5')

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
                        prediction = 'dog'
                        confidence = predictions[0][0]
                        st.write(f'Kết quả dự đoán: {prediction}')
                        st.write(f'Doanh số dự đoán: {confidence * 100:.2f}%')
                    else:
                        prediction = 'cat'
                        confidence = 1 - predictions[0][0]
                        st.write(f'Kết quả dự đoán: {prediction}')
                        st.write(f'Doanh số dự đoán: {confidence * 100:.2f}%')
                
                except Exception as e:
                    st.error(f"Không thể xử lý hình ảnh: {e}")
            else:
                st.write('Xin nhập URL hoặc chuỗi base64 hình ảnh')

            

       
if __name__ == "__main__":
    main()
