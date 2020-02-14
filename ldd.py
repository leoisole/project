# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) '''

# Any results you write to the current directory are saved as output.
import os 
base_dir = os.path.join('..', 'input', 'images_001.zip','images')

all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')

all_xray_df = all_xray_df[0:4999]
len(base_dir)

all_images_path = { os.path.basename(x) : x for x in glob(os.path.join('..', 'input', 'data',  'images_001', '*', '*.png'))}
len(all_images_path)
all_xray_df.shape[0]

all_xray_df['path'] = all_xray_df['Image Index'].map(all_images_path.get)
all_xray_df.sample(5)


#dividing the data into train test.

from sklearn.model_selection import train_test_split 
train_df , test_valid_df = train_test_split(all_xray_df, 
                                   test_size = 0.30, 
                                   random_state = 2018)

valid_df, test_df = train_test_split(test_valid_df, 
                                   test_size = 0.40, 
                                   random_state = 2018)


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel, preprocess_input
from PIL import Image
core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip=False, 
                              vertical_flip=False, 
                              height_shift_range=0.1, 
                              width_shift_range=0.1, 
                              brightness_range=[0.7, 1.5],
                              rotation_range=3, 
                              shear_range=0.01,
                              fill_mode='nearest',
                              zoom_range=0.125,
                             preprocessing_function=preprocess_input)

IMG_SIZE = (299,299)
#Now you can utilize Keras’s ImageDataGenerator to perform image augmentation by directly reading the CSV files through pandas dataframe.
#Takes the dataframe and the path to a directory and generates batches of augmented/normalized data.

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    df_gen = img_data_gen.flow_from_dataframe(in_df,
                                              x_col=path_col,
                                              y_col=y_col,
                                     class_mode = 'raw',
                                    **dflow_args)
    return df_gen


train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'Finding Labels', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 8)



valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'Finding Labels', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 256) 
# we can use much larger batches for evaluation


# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'Finding Labels', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 400))
# one big batch
# used a fixed dataset for final evaluation
final_test_X, final_test_Y = next(flow_from_dataframe(core_idg, 
                               test_df, 
                             path_col = 'path',
                            y_col = 'Finding Labels', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 400)) 

t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
    c_ax.set_title('%s'% ('Finding Labels'))
    c_ax.axis('off')

    
base_pretrained_model = PTModel(input_shape =  t_x.shape[1:], 
                              include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False









