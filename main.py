from flask import Flask, render_template, request
import numpy as np
import keras.models
import re
import sys 
import os
import cv2
import base64
sys.path.append(os.path.abspath("./model"))
from load import * 


CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 103

loaded_model = init(CLASSES)

app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    imagefile = request.files["file"]
    # image_path = "./" + imagefile.filename
    # imagefile.save(image_path)

    # img = cv2.imread(image_path)
    img = np.asarray(bytearray(imagefile.read()), dtype=np.uint8)
    print("img", img.shape)
    img = cv2.imdecode(img,  cv2.IMREAD_COLOR)
    print("img", img.shape)
    img = cv2.resize(img, (224,224))
    input_tensor = np.expand_dims(img, 0)
    input_tensor.shape

    probabilities = loaded_model.predict(input_tensor)
    predictions = np.argmax(probabilities, axis=-1)
    print("Total number of predictions: ", len(predictions))
    print(predictions)

    predictions1 = [CLASSES[predictions[i]] for i in range(len(predictions))]
    print(predictions1[0])

    return f"This is {predictions1[0]} with a probability of {(max(max(probabilities))):.4f}"

if __name__ == '__main__':
    app.run(debug=True, port=8000)


