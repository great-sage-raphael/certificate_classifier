from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model/certificate_cnn.h5')

def predict_certificate(image_path):
    img=image.load_img(image_path, target_size=(128, 128))
    img_array=image.img_to_array(img)/255.0
    img_array= np.expand_dims(img_array,axis=0)

    prediction =model.predict(img_array)[0][0]
    return "Valid Certificate" if prediction > 0.5 else "not a Certificate"

print(predict_certificate('test_Cetificate.jpg'))