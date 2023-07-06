import cv2
from flask import Flask, request, jsonify
import numpy as np
import pickle
import matplotlib.pyplot as plt
# from PIL import Image

model = pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/home')
def index():
    return "Hello world"

def img_to_array(imagefile):
    data=[]
    img = plt.imread(imagefile)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img,[150,150,3])
    # image = Image.fromarray(img)
    # image.save("Image.jpg")
    # img = np.dstack([img, img, img])
    img = img.astype('float32') / 255
    data.append(img)
    return(np.array(data))



@app.route('/predict',methods=['POST'])
def predict():
    xray_image = request.files['xray_image']
    input_query = img_to_array(xray_image)

    result = model.predict(input_query)
    # ans=""
    # if(result>0.7):
    #     ans="Pneumonia"
    # else:
    #     ans="Not Pneumonia"

    return jsonify({'result':str(result)})

if __name__ == '__main__':
    app.run(debug=True)
