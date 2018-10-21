#coding=utf-8
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import input_data
import numpy as np
import model
import os

#get one img from dir
def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    ind = np.random.randint(0,n)
    img_dir = os.path.join(train,files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
	#dir that store test img
    train = 'C:\\Users\\Jason Zhou\\Desktop\\My-TensorFlow-tutorials-master\\cat_and_dog\\testset\\'
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1  # for this program we only read one img, so the batch size would be one
        N_CLASSES = 2  # two output class, which means dog class and cat class
        # Convert image format
        image = tf.cast(image_array, tf.float32)
        # standardize the img
        image = tf.image.per_image_standardization(image)
        # The image was originally three-dimensional [208, 208, 3] redefining the image shape to a 4D four-dimensional tensor
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        # Since the inference returns without an activation function, the result is activated with softmax here.
        logit = tf.nn.softmax(logit)

        # Enter data into the model using the most primitive input data. placeholder
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # the dir that store the module
        logs_train_dir = 'C:\\Users\\Jason Zhou\\Desktop\\My-TensorFlow-tutorials-master\\cat_and_dog\\saveNet\\'
        # define saver
        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Load the model from the path. . . .")
            # load the module into sess
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('The model is loaded successfully, the number of steps in the training is %s' % global_step)
            else:
                print('Model failed to load,,, file not found')
            # calulate the img
            prediction = sess.run(logit, feed_dict={x: image_array})
            # Get the index of the maximum probability in the output(cat or dog)
            max_index = np.argmax(prediction)
            if max_index==0:
                print('the possibilty of cat %.6f' %prediction[:, 0])
            else:
                print('the possibility of dog %.6f' %prediction[:, 1])
# run test
evaluate_one_image()
