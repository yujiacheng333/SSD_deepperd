import tensorflow as tf
import re
import os
from open3d import *
import matplotlib.pyplot as plt
from PIL import Image
from obapi import label_map_util
from obapi import visualization_utils as vis_util
import scipy.misc
NUM_CLASSES = 90
IMAGE_SIZE = (16,9)
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    #return np.array(image.getdata()).reshape(
        #(im_height, im_width, 3)).astype(np.float32)
    pic = np.array(image.getdata())
    print(pic.shape)
    s = pic.shape
    if s[1]==4:
        pic =  pic[:,0:3]
    return pic.reshape(im_height,im_width,3).astype(np.float32)

def get_height_width(y_max,y_min,x_max,x_min,image_size):  # 这里返回二维坐标上的相对距离
    L_X = np.abs(x_max - x_min)
    L_Y = np.abs(y_max - y_min)
    center_X = ((x_max+x_min)/2)
    center_Y = ((y_max+y_min)/2)
    return [L_X,L_Y,center_X,center_Y]
def get_SSD_boxes(train_epo,confident,RGB_image_dir):   ###/models 这种
    cwd = os.getcwd()
    center_dic = {}
    with tf.Session() as sess:
        meta_name = "./models/model-" + str(train_epo) + ".meta"##读取模型
        saver = tf.train.import_meta_graph(meta_name)
        detection_graph = tf.get_default_graph()
        saver.restore(sess, tf.train.latest_checkpoint("./models/"))
        PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        PATH_TO_TEST_IMAGES_DIR = os.getcwd() + RGB_image_dir
        TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
        # TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[1::]
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        rec_depth = detection_graph.get_tensor_by_name("haha/deconv_out/Relu:0")
        # rec_depth = depth_pred
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')   ##读取所有输出张量
        os.chdir(PATH_TO_TEST_IMAGES_DIR)
        img_all_rgb = {}
        for image_path in TEST_IMAGE_PATHS:  # 开始检测
            # print(os.getcwd())
            if image_path == ".DS_Store":
                continue
            image = Image.open(image_path)  # 读图片
            width, height = image.size
            # Actual detection.
            Inimage = image
            image_np = load_image_into_numpy_array(Inimage)
            image_sz = np.array([width,height])
            image_np = np.expand_dims(image_np, axis=0)
            (rec, boxes, scores, classes, num) = sess.run(
                [rec_depth, detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np})

            s_boxes = boxes[scores > confident]
            s_classes = classes[scores > confident]
            s_scores = scores[scores > confident]
            cs_buffer = {}
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
                objinfo_buffer = get_height_width(s_boxes[i][2],s_boxes[i][0],s_boxes[i][3],s_boxes[i][1],image_size=image_sz)
                cs_buffer.update({i:objinfo_buffer})
            # print(rec.shape)
            center_dic.update({image_path.split("/")[-1]:cs_buffer})
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
            img_all_rgb.update({name:image_np.squeeze()/ 256.})
            plt.imshow(image_np.squeeze() / 256.)
            plt.show()
            plt.imshow(rec)  ##之后训练好了把注释取消掉
            plt.show()
        os.chdir(cwd)
    return center_dic,img_all_rgb,   ##这样保证tensorflow只跑一次,一次性读取所有输入图片进行tf 计算
def depth_read(filename):  ##读取深度图train—depth里面有
    img = Image.open(filename).resize([1282, 276], Image.ANTIALIAS)
    # img = Image.open(filename).resize([255, 255],Image.ANTIALIAS)
    depth_png = np.array(img, dtype=np.float32)
    # disp_from_depth = 0.54 * 718.856 / (depth_png+0.01)
    disp_from_depth = depth_png
    disp_from_depth = disp_from_depth.astype(np.float)
    return disp_from_depth
def get3DMAP(center,depth_path,color_path = False,point_sz = 60):  #绘制3维度图
    if color_path == False:
        color_path = os.getcwd()+'/create/lanzi.png'
    #depth_path = os.getcwd() + '/lable_images'+"/b14-9983.png"
    rgb = Image.open(color_path)
    depth = Image.open(depth_path)
    #rgb = rgb.resize([900,500])
    imsz = rgb.size
    c_rec = []
    for c in center:
        c_rec.append([int(c[1]*imsz[1]),int(c[0]*imsz[0])])
    center = c_rec
    #center = [[300, 300], [400, 400]]
    depth = depth.resize(rgb.size)
    depth = 1/(np.array(depth,dtype=np.float32).sum(axis=-1)+0.1)  ##
    scipy.misc.imsave("./create/depth.png",depth)
    rgb.save('./create/rgb.png', 'png')
    ###depth.save('./create/depth.png', 'png')  恢复      恢复       恢复
    color_path = './create/rgb.png'
    depth_path ='./create/depth.png'
    color_raw = read_image(color_path)
    depth_raw = read_image(depth_path)
    #print(np.asarray(color_path).shape)
    rgbd_image = create_rgbd_image_from_color_and_depth(color_raw, depth_raw,depth_trunc = 100.0,convert_rgb_to_intensity = False);
    #print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    pcd = create_point_cloud_from_rgbd_image(rgbd_image, PinholeCameraIntrinsic(
            PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    ##画框图
    pcd_tree = KDTreeFlann(pcd)
    print(center)
    for x in center:
        center_flatten = getFlattenIndex(x,imsz)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[int(center_flatten)], point_sz)
        np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]
    draw_geometries([pcd])
if __name__ == '__main__':
    cwd = os.getcwd()
    center_dic,image_all_rgb = get_SSD_boxes(1,0.5,"/RGB")
    #print(rec.shape)      ==276,1282
    #print(center_dic)
    PATH_TO_depth_IMAGES_DIR = os.getcwd() + '/car_depth'
    depth_path_array = os.listdir(PATH_TO_depth_IMAGES_DIR)
    if '.DS_Store' in depth_path_array:
        depth_path_array.remove('.DS_Store')
    #get3DMAP(pint_sz, center, depth_path, color_path=False)
    for depth_path in depth_path_array:
        PATH_TO_depth_IMAGES_DIR = os.getcwd() + '/car_depth'
        PATH_TO_RGB_IMAGES_DIR = os.getcwd() + "/RGB"
        center = []
        wind = []
        for key in center_dic:
            file_num_RGB = re.sub("\D", "", key)
            #totalCount = re.sub("\D", "", totalCount)
            file_num_dep = re.sub("\D","",depth_path)
            if file_num_dep ==file_num_RGB:
                add_path = key
                for key1 in center_dic[key]:
                    center.append(center_dic[key][key1][2:4])
                    wind.append(center_dic[key][key1][0:2])
                    print("The "+key+"picture's object "+str(key1)+"length is ")
                    print(center_dic[key][key1][0])
                    print("height is ")
                    print(center_dic[key][key1][1])
                    print("\n")
                print(center)
        depth_path = PATH_TO_depth_IMAGES_DIR + "/" + depth_path
        color_path = PATH_TO_RGB_IMAGES_DIR+"/"+add_path
        get3DMAP(center,depth_path,color_path=color_path)
        print(depth_path)