# examples/Python/Tutorial/Basic/rgbd_redwood.py
import os
from open3d import *
import matplotlib.pyplot as plt
from PIL import Image
#def get3DMAP(box_center):
def getFlattenIndex(x,imsz):
    print(imsz[1])
    return imsz[0]*(x[0])+x[1]
if __name__ == '__main__':
    ##get list (np.array)
    center = [[200,400],[200,300]]
    print("Read Redwood dataset")
    color_path = os.getcwd() + '/train_images'+"/b1-99445_Clipped.jpg"
    if color_path == False:
        color_path = os.getcwd()+'/create/lanzi.png'
    depth_path = os.getcwd() + '/lable_images'+"/b14-9983.png"
    rgb = Image.open(color_path)
    depth = Image.open(depth_path)
    #rgb = rgb.resize([900,500])
    imsz = rgb.size
    #center = [[300, 300], [400, 400]]
    depth = depth.resize(rgb.size)
    rgb.save('./create/rgb.png', 'png')
    depth.save('./create/depth.png', 'png')
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
    print(pcd)
    print(imsz[0]*imsz[1])
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    ##画框图
    pcd_tree = KDTreeFlann(pcd)
    for x in center:
        center_flatten = getFlattenIndex(x,imsz)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[center_flatten], 60)
        np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]
    draw_geometries([pcd])