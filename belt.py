import numpy as np
import cv2
import time

INITIAL_ROI = True
FURTHER_ROI = False
CENTER = 995


def cvShow(img, name):
    """
    显示可调控窗口
    :parmam img: 原始图像
    :parmam name: 生成图像窗口名称 
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)


def retinex(image, sigma_list):
    retinex_output = np.zeros_like(image)
    for sigma in sigma_list:
        # 对图像应用高斯模糊
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        image[image == 0] = 1e-10
        blurred[blurred == 0] = 1e-10
        # 计算原始图像和模糊图像的对数差值
        log_scale_diff = np.log(image) - np.log(blurred)

        # 将对数差值添加到输出中
        retinex_output += log_scale_diff

    # 对输出图像进行归一化
    retinex_output = (retinex_output / len(sigma_list)) * 255
    retinex_output = np.uint8(np.clip(retinex_output, 0, 255))

    return retinex_output


def inital_process(img):
    """
    图片进行初步处理
    :param img: 原始图像
    :return: 返回初步处理图像
    """
    # 灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    # 图像增强
    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(10, 10))  # 对图像进行分割,10*10
    gray = clahe.apply(gray)  # 进行直方图均衡化
    # 高斯滤波
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    kernel_2 = np.ones((3, 3), np.uint8)
    # 腐蚀膨胀操作
    erosion = cv2.erode(blur, kernel=kernel_2, iterations=2)
    dilate = cv2.dilate(erosion, kernel_2, iterations=1)

    edges = cv2.Canny(dilate, 250, 180)
    '''
    
    # 改进Retinex增强图像
    sigma_list = [15, 80, 250]
    image = gray.astype(np.float32) / 255.0
    image = retinex(image, sigma_list)
    # 边缘检测
    edges = cv2.Canny(image, 250, 180)
    
    
    return edges


def region_of_interest(img, vertices, sign):
    """
    生成ROI区域
    :param img: 原始图像,是提取了物体边缘的图像
    :param vertices: 多边形坐标
    :return: 返回只保留了ROI区域内的物体边缘的图像
    """
    # 生成和img大小一致的图像矩阵,全部填充0(黑色)
    roi = np.zeros_like(img)
    # vertices即梯形区域顶点,填充梯形内部区域为白色
    if (sign):
        ignore_mask_color = 255
    else:
        roi = cv2.bitwise_not(roi)
        ignore_mask_color = 0
    # 填充函数,将vertices多边形区域填充为指定的灰度值
    cv2.fillPoly(roi, vertices, color=ignore_mask_color)
    # 显示ROI区域
    # cvShow(roi, 'ROI')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 两张图片上的像素进行与操作,ROI区域中已经填充白色,其他是黑色
    # img中是canny算子提取的物体边缘是白色，其他区域是黑色
    # 黑色与黑色与运算还是黑色，白色与白色与运算还是白色，白色与黑色与运算是黑色
    # bitwise_and即两张图片上相同位置像素点上的灰度值进行与运算
    masked_image = cv2.bitwise_and(img, roi)
    # cvShow(masked_image, 'masked_image')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return masked_image


def calculate_slope(line):
    """
    计算斜率
    :param line: np.array([[x_1,y_1,x_2,y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)


def bypass_angle_filter(lines, low_thres, high_thres):
    """
    角度滤波器
    :param lines: 概率霍夫变换得到的直线的端点对集合
    :param low_thres:低阈值
    :param high_thres:高阈值
    :return:得到过滤后的直线端点对集合
    """
    filtered_lines = []
    if lines is None:
        return filtered_lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 过滤掉角度0或90度的直线
            if x1 == x2 or y1 == y2:
                continue
            # 保留角度在low_thres到high_thres之间的直线,角度按360度的标度来算
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            if low_thres < angle < high_thres:
                filtered_lines.append([[x1, y1, x2, y2]])
    return filtered_lines


def reject_abnormal_lines(lines, threshold):
    """
    剔除斜率不一致线段
    :param lines: 线段集合,np.array([[x_1,y_1,x_2,y_2]])
    :return:
    """
    # 计算线段斜率集合
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        # 计算斜率平均值
        mean = np.mean(slopes)
        # 计算斜率与平均值差值集合
        diff = [abs(s - mean) for s in slopes]
        # 索引差值最大值下标
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            # 删除此直线
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines


def least_squares_fit(lines):
    """
    将lines中的线段拟合成一条
    :param lines: 线段集合,np.array([[x_1,y_1,x_2,y_2]])
    :return: 线段上两点,np.array([[xmin,ymin],[xmax,ymax]])
    """
    # 取出所有坐标点
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    # 进行直线拟合,得到多项式系数 k,b
    poly = np.polyfit(x_coords, y_coords, deg=1)
    # 根据多项式系数,计算两个直线上的点，用于唯一确定这条直线 val用于求y = k*x + b
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int64)


def dispose_line(edges):
    """
    处理初步处理后图像生成霍夫线集合
    :param edges: 初步处理图像
    :return: lines 霍夫线集合 np.array([[x_1,y_1,x_2,y_2]])
    """
    # 霍夫变换获取所有线段
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=10, maxLineGap=100)
    # lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=100,
    #                     minLineLength=100, maxLineGap=30)
    return lines


def generate_line(lines):
    """
    生成所需绘制线条
    :param lines: 初步处理图像
    :return: left_line,right_line 所生成两侧直线点集np.array([[x_1,y_1],[x_2,y_2]])
    """
    print('检测出线条点集对总数目:', len(lines))
    lines = bypass_angle_filter(lines, 60, 90)
    print('角度滤波后点集对总数目:', len(lines))
    # 按照斜率划分皮带左右侧
    # 列表生成式
    left_belt = [line for line in lines if calculate_slope(line) < 0]
    print('左侧点集对数目:', len(left_belt))
    right_belt = [line for line in lines if calculate_slope(line) > 0]
    print('右侧点集对数目:', len(right_belt))
    # 过滤线段
    left_belt = reject_abnormal_lines(left_belt, 0.005)
    print('左侧过滤后点集对数目:', len(left_belt))
    right_belt = reject_abnormal_lines(right_belt, 0.005)
    print('右侧过滤后点集对数目:', len(right_belt))
    # 线段拟合
    left_line = least_squares_fit(left_belt)
    right_line = least_squares_fit(right_belt)

    return left_line, right_line


def line_pro_count(img, left_line, right_line):
    """
    图像中延伸并绘制线段
    :param img: 输入图像
    :param left_line: 左侧线段点
    :param right_line: 右侧线段点
    """
    # 取出左右线条点
    left_src_line_1 = tuple(left_line[0])
    left_src_line_2 = tuple(left_line[1])
    right_src_line_1 = tuple(right_line[0])
    right_src_line_2 = tuple(right_line[1])
    # 设置延伸方位
    y_up = 350
    y_down = 900
    y_buttom = 1000
    # 计算左右侧斜率与截距
    left_k = (left_src_line_2[1] - left_src_line_1[1]) / \
        (left_src_line_2[0] - left_src_line_1[0])
    left_b = left_src_line_1[1] - left_k * left_src_line_1[0]
    right_k = (right_src_line_2[1] - right_src_line_1[1]) / \
        (right_src_line_2[0] - right_src_line_1[0])
    right_b = right_src_line_1[1] - right_k * right_src_line_1[0]
    # 计算左侧延长点
    left_x_up = int((y_up - left_b)/left_k)
    left_up = (left_x_up, y_up)
    left_x_down = int((y_down - left_b)/left_k)
    left_down = (left_x_down, y_down)
    left_x_buttom = int((y_buttom - left_b)/left_k)
    # 计算右侧延长点
    right_x_up = int((y_up - right_b)/right_k)
    right_up = (right_x_up, y_up)
    right_x_down = int((y_down - right_b)/right_k)
    right_down = (right_x_down, y_down)
    right_x_buttom = int((y_buttom - right_b)/right_k)

    # 计算皮带中心点坐标
    pixels_between_lines = (left_x_buttom + right_x_buttom)/2

    data = pixels_between_lines - CENTER
    # 判定跑偏并绘制边缘线
    if(data > 0):
        thresh_right = int(pixels_between_lines*0.03)
        if(abs(data) > thresh_right):
            img_show(img,left_up,left_down,False)
            img_show(img,right_up,right_down,False)
            print("右侧跑偏")
        else:
            img_show(img,left_up,left_down,True)
            img_show(img,right_up,right_down,True)
            print("正常")
    else:
        thresh_left= int(pixels_between_lines*0.05)
        if(abs(data) > thresh_left):
            img_show(img,left_up,left_down,False)
            img_show(img,right_up,right_down,False)
            print("左侧跑偏")
        else:
            img_show(img,left_up,left_down,True)
            img_show(img,right_up,right_down,True)
            print("正常")
'''
    # 判定跑偏并绘制边缘线
    if (data > thresh):
        cv2.line(img, left_up, left_down, color=(0, 0, 255), thickness=5)
        cv2.line(img, right_up, right_down, color=(0, 0, 255), thickness=5)
        if((pixels_between_lines - CENTER) > 0):
            print("右侧跑偏")
        else:
            print("左侧跑偏")
    else:
        cv2.line(img, left_up, left_down, color=(0, 255, 0), thickness=5)
        cv2.line(img, right_up, right_down, color=(0, 255, 0), thickness=5)
        print("正常")
'''


def img_show(img, left_line, right_line,sign):
    """
    图像中绘制线段
    :param img: 输入图像
    :param left_line: 左侧线段点
    :param right_line: 右侧线段点
    """
    if(sign):
        cv2.line(img, left_line, right_line, color=(0, 255, 0), thickness=5)
    else:
        cv2.line(img, left_line, right_line, color=(0, 0, 255), thickness=5)
    

if __name__ == '__main__':
    start_time = time.time()
    img = cv2.imread('D:\Project\Python\img\\belt.jpg')
    cv2.waitKey(0)
    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    imgCopy = img.copy()
    edges = inital_process(img)
    vertices = np.array([[[150, 730], [301, 380], [1453, 380], [1618, 730]]])
    masked = region_of_interest(edges, vertices=vertices, sign=INITIAL_ROI)

    # 右侧过滤
    vertices_further_1 = np.array(
        [[[1570, 730], [1500, 540], [1600, 540], [1700, 730]]])
    masked = region_of_interest(
        masked, vertices=vertices_further_1, sign=FURTHER_ROI)

    vertices_further_2 = np.array(
        [[[200, 700], [210, 600], [250, 600], [240, 700]]])
    masked = region_of_interest(
        masked, 
        vertices=vertices_further_2, sign=FURTHER_ROI)
    # 右跑偏过滤
    '''
    vertices_further_3 = np.array(
        [[[220, 670], [262, 405], [370, 405], [330, 670]]])
    masked = region_of_interest(
        masked, vertices=vertices_further_3, sign=FURTHER_ROI)
    '''
    lines = dispose_line(masked)
    left_line, right_line = generate_line(lines)
    line_pro_count(imgCopy, left_line, right_line)
    cvShow(imgCopy, 'img')
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码块运行时间：", execution_time, "秒")

    
    # video = cv2.VideoCapture('top.mp4')
    # # 获取视频的帧速率
    # fps = video.get(cv2.CAP_PROP_FPS)

    # # 设置视频的帧速率
    # video.set(cv2.CAP_PROP_FPS, fps / 2)

    # while (video.isOpened()):
    #     # read()返回两个值 ret为bool frame为视频一帧图像
    #     ret, frame = video.read()
    #     # flip()函数进行图像水平翻转 参数查阅文档
    #     # frame = cv2.flip(frame,-1)
    #     imgCopy = frame.copy()
    #     edges = inital_process(frame)
    #     vertices = np.array([[[150, 730], [301, 380], [1453, 380], [1618, 730]]])
    #     # vertices = np.array([[[220, 730], [300, 240], [1510, 350], [1650, 730]]])
    #     masked = region_of_interest(edges, vertices=vertices,sign=INITIAL_ROI)
    #     vertices_further_1 = np.array([[[1570,730],[1500,540],[1600,540],[1700,730]]])
    #     masked = region_of_interest(masked,vertices=vertices_further_1,sign=FURTHER_ROI)
    #     vertices_further_2 = np.array([[[200,700],[210,600],[250,600],[240,700]]])
    #     masked = region_of_interest(masked,vertices=vertices_further_2,sign=FURTHER_ROI)
    #     lines = dispose_line(masked)
    #     left_line, right_line = generate_line(lines)
    #     line_pro_count(imgCopy, left_line, right_line)
    #     cvShow(imgCopy, 'img')
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    # 释放捕捉对象
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()