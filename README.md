# Machine_learning_API
This project use the CNN module to realize the recognition for cats and dogs.

Before using this API, there are several things you need to do:

* 1.Set up the tensorflow environment.

* 2.Install PIL(pip install pillow)

* 3.Install matplotlib

* 4.Download your own data set

The frame of the program is basicly devided into three parts:

* 1.Image preprocess

* 2.design the nerual networks

* 3.start training

Firstly, the train set should be labeled with their classes(cat & dog). By using the __get_files()__ method, we can save the label and image into the list. In this way, images can be detected by tensorflow.

The nerual network that in this module is CNN, which contains 7 layers:

* conv1:The convolution box is a 3*3 convolution box with a picture thickness of 3 and an output of 16 featuremaps.

* pooling1

* conv2

* pooling2

* local3(fully connected)

* local4(fully connected)

* softmax

By running __training.py__,we can start the trainning process. In this program, we can alter the parameters for the training to fit the trainset. If epoch is too big, the result may lead to overfitting.

## Result
According to the test result, the model can successfully recognize the difference of cats and dogs. But there are some situations that it failed to recognize it. I consider that the main reason is that the training set is small. I tried several times with different training steps to get a better result and finally get a idealized result by training in 4k steps with 64 images per batch.

<img src= "https://github.com/JasonZ95/Machine_learning_API/blob/master/0d82bf2b2cded83203edf2b5236b32f.jpg" />
<img src= "https://github.com/JasonZ95/Machine_learning_API/blob/master/30d0bb2cc2de381102d08140f2333fc.png" />
<img src= "https://github.com/JasonZ95/Machine_learning_API/blob/master/397599579527309947.jpg" />
<img src= "https://github.com/JasonZ95/Machine_learning_API/blob/master/a75aa22366e33b7ad545c341b7fda58.png" />

