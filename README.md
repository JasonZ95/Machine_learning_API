# Machine_learning_API
This project use the CNN module to realize the recognition for cats and dogs.

Before using this API, there are several things you need to do:

1.Set up the tensorflow environment.

2.Install PIL(pip install pillow)

3.Install matplotlib

4.Download your own data set

The frame of the program is basicly devided into three parts:

1.Image preprocess

2.design the nerual networks

3.start training

Firstly, the train set should be labeled with their classes(cat & dog). By using the __get_files()__ method, we can save the label and image into the list. In this way, images can be detected by tensorflow.

The nerual network that in this module is CNN, which contains 7 layers:

#conv1:The convolution box is a 3*3 convolution box with a picture thickness of 3 and an output of 16 featuremaps.

#pooling1

#conv2

#pooling2

#local3(fully connected)

#local4(fully connected)

#softmax

By running __training.py__,we can start the trainning process. In this program, we can alter the parameters for the training to fit the trainset. If epoch is too big, the result may lead to overfitting.
