import sys 
import os 
from tensorflow.python.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.layers import Convultion2D, MaxPooling20
from tensorflow.python.keras import backend as K

K.clear_session()

data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

## Parametros
epocas=20
altura, longitud = 100, 100
batch_size=32
pasos=1000
pasos_validacion=200
filtrosConv1=32
filtrosConv2=64
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=5
lr=0.0005

## pre procesamiento de imagenes
entrenamiento_datagen = ImageDataGenerator(
  rescale = 1./255,
  shear_range = 0.3,
  zoom_range = 0.3,
  horizontal_flip = True
) 

validacion_datagen = ImageDataGenerator(
  rescale=1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
  data_entrenamiento,
  target_size=(altura, longitud),
  batch_size=batch_size,
  class_mode='categorical'
)

imagen_validacion =  validacion_datagen.flow_from_directory(
  data_validacion
)