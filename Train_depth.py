import numpy as np
import os
import tensorflow as tf
import random
from PIL import Image
from obapi import label_map_util
from mpl_toolkits.mplot3d import Axes3D
from obapi import visualization_utils as vis_util
batch_size = 20
train_epo = 1
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
IMAGE_SIZE = (16, 9)
confident = 0.2
NUM_CLASSES = 90  # 检测对象个数
def depth_read(filename):
    print(Image.open(filename).size)
    img = Image.open(filename).resize([1282,276],Image.ANTIALIAS)
    #img = Image.open(filename).resize([255, 255],Image.ANTIALIAS)
    depth_png = np.array(img, dtype=np.float32)
    #disp_from_depth = 0.54 * 718.856 / (depth_png+0.01)
    disp_from_depth = depth_png
    #disp_from_depth[depth_png < 0] = -1
    #assert np.max(disp_from_depth) > 255,"np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)
    disp_from_depth = disp_from_depth.astype(np.float)
    #depth[depth_png == 0] = -1.
    return disp_from_depth
def get_random_batch(size):
    return np.random.randint(0,1,[size,255,255,3]),np.random.randint(0,1,[size,255,255,1])
def getsiglebatch():
    in_batch = []
    now_path = os.getcwd()
    lable_batch = []
    PATH_TO_Train_IMAGES_DIR = os.getcwd() + '/train_images'
    PATH_TO_depth_IMAGES_DIR = os.getcwd() + '/lable_images'
    Input_path_array = os.listdir(PATH_TO_Train_IMAGES_DIR)
    lable_path_array = os.listdir(PATH_TO_depth_IMAGES_DIR)
    if '.DS_Store' in Input_path_array:
        Input_path_array.remove('.DS_Store')
        # print("垃圾苹果1")
    if '.DS_Store' in lable_path_array:
        lable_path_array.remove('.DS_Store')
        # print("垃圾苹果2")
    input = Input_path_array[10]
    lable = 0
    for lable_file_name in lable_path_array:
        if lable_file_name.split(".")[0]  in input:
            lable = lable_file_name
    #print(lable)
    #print(input)
    # for file_name in input:
    #    for lable_file_name in lable_path_array:
    #        namearray = file_name.split("_")
    #        if namearray[-3] in lable_file_name and namearray[-1] in lable_file_name:
    #            lable.append(lable_file_name)
    #            break
    os.chdir(PATH_TO_Train_IMAGES_DIR)
    print(PATH_TO_Train_IMAGES_DIR)
    Inimage = Image.open(input)
    image_np = load_image_into_numpy_array(Inimage)
        # print(image_np.shape)
    in_batch.append(image_np)
    os.chdir(PATH_TO_depth_IMAGES_DIR)
    deepimg = depth_read(lable)
    deepimg = np.sum(deepimg,axis=-1)
    deepimg = np.expand_dims(deepimg, axis=-1)
    print(deepimg.shape)
    #deepimg = np.squeeze(deepimg)
    lable_batch.append(deepimg)
    os.chdir(now_path)
    return in_batch, lable_batch
def getbatch(size):
    in_batch = []
    now_path = os.getcwd()
    lable_batch = []
    PATH_TO_Train_IMAGES_DIR = os.getcwd() + '/train_images'
    PATH_TO_depth_IMAGES_DIR = os.getcwd() + '/lable_images'
    Input_path_array = os.listdir(PATH_TO_Train_IMAGES_DIR)
    lable_path_array = os.listdir(PATH_TO_depth_IMAGES_DIR)
    if '.DS_Store' in Input_path_array:
        Input_path_array.remove('.DS_Store')
        # print("垃圾苹果1")
    if '.DS_Store' in lable_path_array:
        lable_path_array.remove('.DS_Store')
        # print("垃圾苹果2")
    input = random.sample(Input_path_array,size)
    lable = []
    for file_name in input:
        for lable_file_name in lable_path_array:
            if lable_file_name.split(".")[0] in file_name:
                lable.append(lable_file_name)
    print(lable)
    print(input)
    #for file_name in input:
    #    for lable_file_name in lable_path_array:
    #        namearray = file_name.split("_")
    #        if namearray[-3] in lable_file_name and namearray[-1] in lable_file_name:
    #            lable.append(lable_file_name)
    #            break
    os.chdir(PATH_TO_Train_IMAGES_DIR)
    for image_path in input:
        Inimage = Image.open(image_path)
        Inimage = Inimage.resize([1282,276],Image.ANTIALIAS)
        image_np = load_image_into_numpy_array(Inimage)
        # print(image_np.shape)
        in_batch.append(image_np)
    os.chdir(PATH_TO_depth_IMAGES_DIR)
    for laimage in lable:
        deepimg = depth_read(laimage)
        #deepimg = np.expand_dims(deepimg, axis=-1)
        deepimg = np.sum(deepimg,axis=-1)
        deepimg = np.expand_dims(deepimg, axis=-1)
        lable_batch.append(deepimg)
    lable_batch = np.array(lable_batch)
    os.chdir(now_path)
    return in_batch, lable_batch
def getTrainedModel():
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    #PATH_TO_TEST_IMAGES_DIR = os.getcwd() + '/test_images'
    #os.chdir(PATH_TO_TEST_IMAGES_DIR)
    return detection_graph
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.float32)

    #return {"x":1,"y":2}
def deconvalayers(detection_graph):
    cwd = os.getcwd()
    with detection_graph.as_default():
        with tf.name_scope("haha"):
            convtensor = detection_graph.get_tensor_by_name("FeatureExtractor/MobilenetV2/Conv_1/Relu6:0")
            image_tensor1 = detection_graph.get_tensor_by_name(
                "FeatureExtractor/MobilenetV2/expanded_conv_13/output:0")  ##size>10 feature mapssss from mobile net V2
            image_tensor2 = detection_graph.get_tensor_by_name(
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_2_3x3_s2_512/Relu6:0")
            image_tensor3 = detection_graph.get_tensor_by_name(
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_3_3x3_s2_256/Relu6:0")
            image_tensor4 = detection_graph.get_tensor_by_name(
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_4_3x3_s2_256/Relu6:0")
            image_tensor5 = detection_graph.get_tensor_by_name(
                "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_5_3x3_s2_128/Relu6:0")
            #print(image_tensor1)
            print(image_tensor1)##10 10 160
            print(image_tensor2)##5 5 512
            print(image_tensor3)##3 3 256
            print(image_tensor4)##2 2 256
            print(convtensor)#10 10 1280
            image_tensor5 = tf.layers.conv2d_transpose(inputs=image_tensor5,strides=(1,1),kernel_size=(2,2), filters=1024
                                                       , activation="relu", padding="valid",kernel_initializer=
                                                       tf.truncated_normal_initializer(stddev=0.01,mean=0.1),
                                                       data_format="channels_last",)
            image_tensor5 = batch_norm(image_tensor5, is_training=True)
            image_tensor4 = tf.concat([image_tensor4,image_tensor5],axis=-1)
            image_tensor4 = tf.layers.conv2d_transpose(inputs=image_tensor4,strides=(1,1),kernel_size=(2,2), filters=1024
                                                       , activation="relu", padding="valid",kernel_initializer=
                                                       tf.truncated_normal_initializer(stddev=0.01,mean=0.1),
                                                       data_format="channels_last",)
            image_tensor4 = batch_norm(image_tensor4, is_training=True)
            image_tensor3 = tf.concat([image_tensor4,image_tensor3],axis=-1)
            image_tensor3 = tf.layers.conv2d_transpose(inputs=image_tensor3,strides=(1,1),kernel_size=(3,3), filters=1024
                                                       , activation="relu", padding="valid",kernel_initializer=
                                                       tf.truncated_normal_initializer(stddev=0.01,mean=0.1),
                                                       data_format="channels_last",)
            image_tensor3 = batch_norm(image_tensor3, is_training=True)
            image_tensor2 = tf.concat([image_tensor3, image_tensor2], axis=-1)
            image_tensor2 = tf.layers.conv2d_transpose(inputs=image_tensor2,strides=(2,2),kernel_size=(2,2), filters=1024
                                                       , activation="relu", padding="valid",kernel_initializer=
                                                       tf.truncated_normal_initializer(stddev=0.01,mean=0.1),
                                                       data_format="channels_last")
            image_tensor2 = batch_norm(image_tensor2, is_training=True)
            image_out_tensor = tf.concat([image_tensor2,image_tensor1,convtensor],axis=-1)
            #print(image_out_tensor,is_training=True)
            image_out_tensor = batch_norm(image_out_tensor, is_training=True)
            image_out_tensor = tf.layers.conv2d(inputs=image_out_tensor,kernel_size=(3,3),filters=648,strides=(1,1),padding = "same",activation="relu"
                                                ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0.1),)
            image_out_tensor = batch_norm(image_out_tensor, is_training=True)
            image_out_tensor = tf.layers.conv2d(inputs=image_out_tensor,kernel_size=(3,3),filters=256,strides=(1,1),padding = "same",activation="relu"
                                                ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0.1),)
            image_out_tensor = batch_norm(image_out_tensor,is_training=True)
            image_out_tensor = tf.layers.conv2d(inputs=image_out_tensor,kernel_size=(3,3),filters=256,strides=(1,1),padding = "same",activation="relu"
                                                ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01,mean=0.1),)
            image_out_tensor = batch_norm(image_out_tensor, is_training=True)
            depth_pred = tf.layers.conv2d_transpose(inputs=image_out_tensor,strides=(3,3),filters=256,kernel_size=[4,7],padding = "valid"
                                                ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01,mean= 0.1), activation="relu")
            depth_pred = batch_norm(depth_pred, is_training=True)
            depth_pred = tf.layers.conv2d_transpose(inputs=depth_pred,strides=(2,2),filters=256,kernel_size=[4,4],padding = "valid"
                                                ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation="relu")
            depth_pred = batch_norm(depth_pred, is_training=True)
            depth_pred = tf.layers.conv2d_transpose(inputs=depth_pred, strides=(1, 2), filters=128,
                                                    kernel_size=[4, 4], padding="valid"
                                                    , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),activation="relu" )
            depth_pred = batch_norm(depth_pred, is_training=True)
            depth_pred = tf.layers.conv2d_transpose(inputs=depth_pred, strides=(2, 3), filters=128,
                                                    kernel_size=[5, 4], padding="valid"
                                                    , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),activation="relu")
            depth_pred = batch_norm(depth_pred, is_training=True)
            depth_pred = tf.layers.conv2d_transpose(inputs=depth_pred, strides=(2, 3), filters=1,
                                                    kernel_size=[4, 4], padding="valid",activation="relu"
                                                    , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),bias_initializer=tf.truncated_normal_initializer(mean=0.1,stddev=0.01),name="deconv_out")

            # y = tf.placeholder(shape=[X.shiape],dtype=tf.float32)
            # M_Graph = tf.Graph()
            current_iter = tf.Variable(0)
            #y = tf.placeholder(shape=[None,255, 255, 1], dtype=tf.float32)
            X = detection_graph.get_tensor_by_name('image_tensor:0')
            y = tf.placeholder(shape = [None,276,1282,1],dtype=tf.float32)
            #l = 0.9*tf.nn.l2_loss(depth_pred-y)+0.01*tf.reduce_sum(tf.abs(depth_pred))
            l = tf.reduce_sum (tf.square(tf.log(depth_pred+1)-tf.log(y+1)))
            #l = tf.reduce_mean(tf.abs((tf.log(y+1)-tf.log(depth_pred+1))))
            #l = tf.sqrt(tf.reduce_mean(tf.log(tf.square(y))-tf.log(tf.square(depth_pred))))
            #c = tf.train.exponential_decay(learning_rate=0.1, decay_steps=5, decay_rate=0.92,global_step=current_iter)
            #l=tf_L(y,depth_pred, valid_pixels=True, gamma=0.5)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(l)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                #Saver.restore(sess,"../")
                #step = 0
                #summary_writer = tf.summary.FileWriter("log", sess.graph)
                ckpt = tf.train.get_checkpoint_state("./models")
                if ckpt and ckpt.model_checkpoint_path:  ##判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练
                    print("++++++++++++restore_from_saver++++++++++++++")
                    saver.restore(sess, ckpt.model_checkpoint_path)
                for i in range(train_epo):
                    current_iter = i
                    #batch_xs, batch_ys = getsiglebatch()
                    batch_xs, batch_ys =getbatch(batch_size)
                    batch_xs = np.array(batch_xs)
                    batch_ys = np.array(batch_ys)
                    print(batch_xs.shape)
                    print(batch_ys.shape)
                    batch_ys = batch_ys/256.
                    batch_ys = batch_ys
                    [Loss,_] = sess.run([l,train_step],feed_dict={X:batch_xs,y:batch_ys})
                    print(batch_xs.shape)
                    print(batch_ys.shape)
                    if i%100==0:
                        plt.imshow(batch_xs[0,:,:,:])
                        plt.show()
                        plt.imshow(batch_ys[0,:,:,:].squeeze())
                        plt.show()
                        plt.imshow(sess.run(depth_pred,feed_dict={X:batch_xs})[0,:,:,:].squeeze())
                        plt.show()
                    saver.save(sess, "models/model", global_step=train_epo)
                    print("Train step_______  "+str(i)+"Loss is _____  "+str(Loss))
                summary_writer = tf.summary.FileWriter("log", sess.graph)
                #saver = tf.train.Saver(max_to_keep=0)
                saver.save(sess,"models/model",global_step=train_epo)

                #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,[''])
                #with tf.gfile.FastGFile("./" + 'model.pb', mode='wb') as f:
                 #   f.write(constant_graph.tostring())
                    #f.write(constant_graph.SerializeToString())
                os.chdir(cwd)
                return detection_graph
                #saver = tf.train.Saver(max_to_keep=0)
def test(train_epo):
    cwd = os.getcwd()
    with tf.Session() as sess:
        meta_name = "./models/model-"+str(train_epo)+".meta"
        saver = tf.train.import_meta_graph(meta_name)
        detection_graph = tf.get_default_graph()
        saver.restore(sess, tf.train.latest_checkpoint("./models/"))
        PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                        use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        PATH_TO_TEST_IMAGES_DIR = os.getcwd() + '/test_images'
        TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
            #TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[1::]
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        rec_depth = detection_graph.get_tensor_by_name("haha/deconv_out/Relu:0")
            #rec_depth = depth_pred
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        os.chdir("./test_images")
        for image_path in TEST_IMAGE_PATHS:  # 开始检测
            #print(os.getcwd())
            if image_path ==".DS_Store":
                continue
            image = Image.open(image_path)  # 读图片
            width, height = image.size
                # Actual detection.
            Inimage = image
            image_np = load_image_into_numpy_array(Inimage)
            image_np = np.expand_dims(image_np,axis=0)
            (rec,boxes, scores, classes, num) = sess.run(
                [rec_depth,detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np})

            s_boxes = boxes[scores > confident]
            s_classes = classes[scores > confident]
            s_scores = scores[scores > confident]
            print(s_classes)
            for i in range(len(s_classes)):
                name = image_path.split("/")[-1]
                # name = image_path.split("\\")[-1].split('.')[0]   # 不带后缀
                ymin = s_boxes[i][0] * height  # ymin
                xmin = s_boxes[i][1] * width  # xmin
                ymax = s_boxes[i][2] * height  # ymax
                xmax = s_boxes[i][3] * width  # xmax
                score = s_scores[i]
                if s_classes[i] in category_index.keys():
                    class_name = category_index[s_classes[i]]['name']
                print("name:", name)
                print("ymin:", ymin)
                print("xmin:", xmin)
                print("ymax:", ymax)
                print("xmax:", xmax)
                print("score:", score)
                print("class:", class_name)
                print("################")
            #print(rec.shape)
            rec = rec.squeeze()
            plt.figure(figsize=IMAGE_SIZE)
            vis_util.visualize_boxes_and_labels_on_image_array(
            np.squeeze(image_np),
            np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1)

            plt.imshow(image_np.squeeze()/256.)
            plt.show()
            #plt.imshow(rec)  ##depthestimation
            #plt.show()
                #point_3D(rec)##3Drec
        os.chdir(cwd)

LOG_INITIAL_VALUE= 1
def tf_L(tf_y, tf_y_, valid_pixels=True, gamma=0.5):
    loss_name = "Scale Invariant Logarithmic Error"

    tf_log_y  = tf.log(tf_y  + LOG_INITIAL_VALUE)
    tf_log_y_ = tf.log(tf_y_ + LOG_INITIAL_VALUE)

    # Calculate Difference and Gradients. Compute over all pixels!
    tf_d = tf_log_y - tf_log_y_
    tf_gx_d = gradient_x(tf_d)
    tf_gy_d = gradient_y(tf_d)

    # Mask Out
    if valid_pixels:
        # Identify Pixels to be masked out.
        tf_idx = tf.where(tf_y_ > 0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

        # Overwrites the 'd', 'gx_d', 'gy_d' tensors, so now considers only the Valid Pixels!
        tf_d = tf.gather_nd(tf_d, tf_idx)
        tf_gx_d = tf.gather_nd(tf_gx_d, tf_idx)
        tf_gy_d = tf.gather_nd(tf_gy_d, tf_idx)

    # Loss
    tf_npixels = tf.cast(tf.size(tf_d), tf.float32)
    mean_term = (tf.reduce_sum(tf.square(tf_d)) / tf_npixels)
    variance_term = ((gamma / tf.square(tf_npixels)) * tf.square(tf.reduce_sum(tf_d)))
    grads_term = (tf.reduce_sum(tf.square(tf_gx_d)) + tf.reduce_sum(tf.square(tf_gy_d))) / tf_npixels

    tf_loss_d = mean_term - variance_term + grads_term

    return loss_name, tf_loss_d
def gradient_x(img):
    gx = img[:, :, :-1] - img[:, :, 1:]

    # Debug
    # print("img:", img.shape)
    # print("gx:",gx.shape)

    return gx


def gradient_y(img):
    gy = img[:, :-1, :] - img[:, 1:, :]

    # Debug
    # print("img:", img.shape)
    # print("gy:",gy.shape)

    return gy
def point_3D(matix):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 255, 1)
    Y = np.arange(0, 255, 1)
    X, Y = np.meshgrid(X, Y)
    ax.view_init(elev=130., azim=0)
    ax.plot_surface(X, Y, matix, rstride=10, cstride=10, cmap='rainbow')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #plt.show()
def batch_norm(inputs, is_training,  epsilon = 0.001, decay = 0.99):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)
if __name__ == '__main__':
    var_test = 1
    latest_train_epo = train_epo
    if var_test:
        test(latest_train_epo)
    else:
        graph = getTrainedModel()
        deconvalayers(graph)