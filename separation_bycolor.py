import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, draw, color
dic = {"lveavs_and_grass": ([26, 43, 46], [99,255,255])}
file_name = "./test_images/4.tif"


def conv_2D_valid(X, kernel=np.ones([3,3])):
    res = np.zeros(X.shape)
    l = kernel.shape[0]
    w = kernel.shape[1]
    X = np.pad(X, ((kernel.shape[0], kernel.shape[0]), (kernel.shape[1], kernel.shape[1])), "constant", constant_values=(0, 0))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            ct_x = i + kernel.shape[0]
            ct_y = j + kernel.shape[1]
            res[i, j] = np.sum(X[ct_x-int(l/2):(ct_x+(l-int(l/2))), ct_y-int(w/2):(ct_y+(w-int(w/2)))] * kernel)
    return res


def sep_cutoff_dese_area(X, stride):
    leak_l = X.shape[0] % stride
    leak_w = X.shape[1] % stride
    X = np.pad(X, ((0, leak_l), (0, leak_w)), "constant", constant_values=(0, 0))
    for i in range(int(X.shape[0]/stride)):
        for j in range(int(X.shape[1]/stride)):
            buffer = X[i*stride: (i+1)*stride,j*stride:(j+1)*stride]
            if np.mean(buffer)>0.5:
                X[i * stride: (i + 1) * stride, j * stride:(j + 1) * stride] = 0
    return X


def cutoff_green_area(X, stride, img):
    leak_l = X.shape[0] % stride
    leak_w = X.shape[1] % stride
    mean_var = np.mean(img[:, :, 1])
    X = np.pad(X, ((0, leak_l), (0, leak_w)), "constant", constant_values=(0, 0))
    for i in range(int(X.shape[0] / stride)):
        for j in range(int(X.shape[1] / stride)):
            buffer = img[i * stride: (i + 1) * stride, j * stride:(j + 1) * stride, :]

            if np.mean(buffer[:, :, 1])>mean_var:
                X[i * stride: (i + 1) * stride, j * stride:(j + 1) * stride] = 0
    return X


def get_diff_graph(X, mode ="Sobel"):
    if mode == "Sobel":
        Gx = np.asmatrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 0]])
        Gy = np.asmatrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    if mode == "Prewitt":
        Gx = np.asmatrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy = np.asmatrix([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    outer_Gx = conv_2D_valid(X, Gx)
    outer_Gy = conv_2D_valid(X, Gy)
    G = np.sqrt(outer_Gx**2 + outer_Gy**2)
    seta = np.arctan(outer_Gy / (outer_Gx+1e-10))
    return G, seta


def round_local(x):
    res = x - int(x)
    if res < 0.5:
        return int(x)
    else:
        return int(x)+1


def cacu_concat_area(x_pos, y_pos, mode="self_routed"):

    if len(x_pos) < 100:
        return False
    elif mode == "Person":
        covxy = np.cov(x_pos, y_pos)
        rou = np.cov(x_pos, y_pos)[0, 1] / np.sqrt(covxy[0, 0] * covxy[1, 1])
        if rou < 0.5:
            return True
        else:
            return False
    elif mode == "linear_err":
        mean_y = np.mean(y_pos)
        mean_x = np.mean(x_pos)
        k = np.cov(x_pos, y_pos)[0, 1]/np.cov(x_pos, x_pos)[0, 0]
        b = np.mean(mean_y - k*mean_x)
        y_hat = k*x_pos + b
        dist_sum = np.mean(np.abs(y_pos - y_hat))  # MAE 如果想用均方误差就加**2就完事了
        if dist_sum < 0.4:
            return True
        else:
            return False
    elif mode == "self_routed":
        shape = [int(np.max(x_pos))+2, int(np.max(y_pos))+2]
        zone = np.zeros(shape)
        for i in range(len(x_pos)):
            zone[round_local(x_pos[i]), round_local(y_pos[i])] = 1
        count_x = 0
        count_y = 0
        scanned_x = []
        scanned_y = []
        for i in range(len(x_pos)):
            x_idx = round_local(x_pos[i])
            if np.sum(zone[x_idx, :]) >= 2:
                count_x += 1
            y_idx = round_local(y_pos[i])
            if np.sum(zone[:, y_idx]) >= 2:  # and y_idx not in scanned_y 不加入重复检测，
                # 这样穿过多次的点会被重复计数而曲线不平滑的线会多次遍历 实验结果而已，没有算法
                count_y += 1
            scanned_x.append(x_pos[i])
            scanned_y.append(y_pos[i])
        count_x /= len(x_pos)
        count_y /= len(x_pos)
        count = np.min(np.asarray([count_y, count_x]))
        if count < 0.9:
            return True
        else:
            return False


def local_desort(x):

    res = [x[x.shape[0] - i -1] for i in x]
    return np.asarray(res)


def expand_area_local(img, pts_buffer):
    mask_out = []
    l = img.shape[0]
    w = img.shape[1]
    mask = np.ones([l, w], dtype=np.uint8)
    count = 1
    for pts in pts_buffer:
        for j in range(len(pts[0])):
            mask[round_local(pts[1][j]), round_local(pts[0][j])] = 0
    start_point = []
    stride = 8
    stride_x = int(l/stride)
    stride_y = int(w/stride)
    for q in range(stride+2):
        for r in range(stride+2):
            start_point.append((q*stride_x, r*stride_y))

    for i in range(stride**2):

        init_x = start_point[i][0]
        init_y = start_point[i][1]
        if init_x >= l:
            init_x = l-5
        if init_y >= w:
            init_y = w-5
        if mask_out!=[]:
            sum_map = np.sum(np.asarray(mask_out), axis=0)
        else:
            sum_map = np.zeros([img.shape[0], img.shape[1]])
        if sum_map[init_x, init_y] != 1:
            mask_cp = mask.copy()
            cv2.floodFill(mask_cp, np.zeros([mask.shape[0]+2, mask.shape[1]+2], dtype=np.uint8), (init_x, init_y),(0, 0, 0)
                              , cv2.FLOODFILL_FIXED_RANGE)
            mask_cp = mask_cp==0
            if mask_out == [] and np.sum(mask_cp)<0.5*l*w and np.sum(mask_cp)>0.2*l*w:
                mask_out.append(mask_cp)
                plt.imshow(mask_cp)
                plt.show()
            else:
                f = True
                if np.sum(mask_cp)<0.8*l*w or np.sum(mask_cp)>0.1*l*w:
                    for mm in mask_out:
                        print(np.sum(np.abs(mm - (mask_cp).astype(int))))
                        if np.sum(np.abs(mm - (mask_cp).astype(int)))<0.1*l*w:
                            f = False
                            break
                    if f:
                        mask_out.append(mask_cp)
                        plt.imshow(mask_cp)
                        plt.show()
        xx = np.zeros([img.shape[0], img.shape[1]])
        for x in mask_out:
            xx+=x


    return count



def color_sepration(filename, color_map):
    pts_buffer = []
    img = cv2.imread(filename)
    img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
    res = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.sum(img[i, j, :]) == 0:
                res[i, j] = 1
    contours = measure.find_contours(res, 0.6)
    max_len = 0
    mean_len = 0
    for n, contour in enumerate(contours):
        buffer = len(contour[:, 1])
        if buffer>max_len:
            max_len = buffer
        mean_len+=buffer
    mean_len /= n
    for n, contour in enumerate(contours):
        if not cacu_concat_area(contour[:, 1], contour[:, 0]):
            continue
        if len(contour[:, 1])>max_len*0.2:
            pts_buffer.append([contour[:, 1], contour[:, 0]])
            # plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    img = cv2.imread(filename)
    # 纯纹理分割结束
    # 颜色反选
    img = cv2.GaussianBlur(img, (3, 3), 0)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = []
    for (lower, upper) in color_map:
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        mask.append(cv2.inRange(HSV, lower, upper))

    for j in range(len(mask)):
        mask[j] = (mask[j] / 255).astype(bool)
    for m in mask:
        if np.sum(m) > 1000:
            contours = measure.find_contours(m, 0.9)
            max_len = 0
            mean_len = 0
            for n, contour in enumerate(contours):
                buffer = len(contour[:, 1])
                if buffer > max_len:
                    max_len = buffer
                mean_len += buffer
            mean_len /= n
            for n, contour in enumerate(contours):
                if not cacu_concat_area(contour[:, 1],contour[:, 0]):
                    continue
                pts_buffer.append([contour[:, 1], contour[:, 0]])
                #if len(contour[:, 1]) > max_len * 0.6:
                    #plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            img = cv2.imread(filename)


            #for pts in pts_buffer:
                #plt.plot(pts[0], pts[1], linewidth=2)
            count = expand_area_local(img, pts_buffer)



if __name__ == '__main__':

    color = []
    for i in dic:
        color.append(dic[i])
    color_sepration(file_name, color)
