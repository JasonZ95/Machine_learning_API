import os
import numpy as np
import tensorflow as tf
import input_data
import model


N_CLASSES = 2  #2 output neurons,［1，0］ or［0，1］the possibilty of cat and dog
IMG_W = 208  # Redefine the size of the image. If the image is too large, the training will be slow.
IMG_H = 208
BATCH_SIZE = 32  #The size of each batch
CAPACITY = 256
MAX_STEP = 6000 # epoch=step*batch_size
learning_rate = 0.00008


def run_training():

    # 数据集
    train_dir = 'C:\\Users\\Jason Zhou\\Desktop\\My-TensorFlow-tutorials-master\\cat_and_dog\\img\\'   #My dir
    #logs_train_dir: The data of the process of storing the training model, viewed in the tensorboard
    logs_train_dir = 'C:\\Users\\Jason Zhou\\Desktop\\My-TensorFlow-tutorials-master\\cat_and_dog\\saveNet\\'

    # get the label and img
    train, train_label = input_data.get_files(train_dir)
    # generate batch
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    # enter the module
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    # get train loss
    train_loss = model.losses(train_logits, train_label_batch)
    # train
    train_op = model.trainning(train_loss, learning_rate)
    # get accuracy
    train__acc = model.evaluation(train_logits, train_label_batch)
    # summary
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    # save summary
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                # Save the model every 2000 steps and save the model in checkpoint_path
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

# train
run_training()
