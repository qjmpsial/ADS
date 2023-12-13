import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('final_model.h5')
  return model
model=load_model()
st.write("""
# Image Classifier"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
# from PIL import Image,ImageOps
from PIL import Image,ImageOps,ImageFilter
import numpy as np

def import_and_predict(image_data,model):
    # size=(64,64)
    size=(32,32)
    # image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Use Image.LANCZOS for antialiasing
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

# def import_and_predict(image_data, model):
    # size=(64,64)
#     size = (32, 32)
#     # image = image_data.resize(size, Image.ANTIALIAS)  # <-- Modified line
#     image = ImageOps.fit(image_data, size, ImageFilter.ANTIALIAS) # <-- Modified line
#     img = np.asarray(image)
#     img_reshape = img[np.newaxis, ...]
#     prediction = model.predict(img_reshape)
#     return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '15', '16',
       '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
       '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
       '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
       '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84',
       '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
