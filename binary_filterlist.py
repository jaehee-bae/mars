# 이진화 필터 리스트

import cv2
import numpy as np
import scipy.ndimage
import utils as util

# =========
# 필터 적용 시 필요한 함수 정의
def grayscale(rgb):
	return np.dot(rgb[...,:3],[0.299,0.587,0.114])

def dodge(front,back):
	result=front*255/(255-back)
	result[result>255]=255
	result[back==255]=255
	return result.astype('uint8')

def get_difference(image1, image2):

    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    image1[mask != 255] = [0, 0, 255]
    # image2[mask != 255] = [0, 0, 255]

    # image1에서 빨간색만 찾아서 남기기
    final_img = util.make_black_img(image1, 255, 0, 0)

    return final_img
    # final_img.show()
    # cv2.imshow("image1", image1)

def canny(img):
    img = cv2.Canny(img, 170, 200)
    return img

# MSRCP
def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img).astype(np.float64)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img

def feature_filter(img, d, sigma_c, sigma_s):

    ## 1. (원본 이미지) - (쌍방향 필터 적용 이미지)
    # d, sigmaColor 값이 클수록 노이즈가 제거되지만 디테일한 표현이 떨어짐
    # 눈 : 15, 200, 200 (속눈썹 표현, 노이즈 제거)
    # 코 : 20 or 25(노이즈 덜함), 100, 100 (콧볼, 노이즈 제거 d=25 일 때 노이즈가 완전 없지만 디테일이 떨어짐)
    # 눈썹 : 25, 100, 100
    # 입 : 20, 100, 100
    bifilter_img = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_c, sigmaSpace=sigma_s)
    bifilter_img_pil = get_difference(bifilter_img, img)

    bifilter_img = np.array(bifilter_img_pil)

    ## 3. Sketch Filter + Canny
    # 눈 : sigma = 100, 엣지는 잘 잡아서 1번과 합쳐서 사용
    # 코 : sigma = 100, 콧볼을 너무 못잡아서 가망 없음 1번과 합쳐서 사용
    # 눈썹 : sigma = 100, 1번과 합쳐서 사용
    gray_img = grayscale(img)
    i = 255 - gray_img
    b = scipy.ndimage.filters.gaussian_filter(i, sigma=100)  # 딱히 sigma 값이 의미가 없음
    sketch_img = dodge(b, gray_img)
    sketch_canny_img = canny(sketch_img)

    # 이미지 색상 반전 (흰색->검은색, 검은색->흰색)
    sketch_canny_img = cv2.bitwise_not(sketch_canny_img)

    ## (최종 선택한 필터) 4. 필터 조합 - 쌍방 필터 + canny
    # 연산을 위해서 채널 변경 (4->1?)
    bifilter_img = cv2.cvtColor(bifilter_img, cv2.COLOR_RGBA2GRAY)

    bit_and = cv2.bitwise_and(bifilter_img, sketch_canny_img)

    return bit_and


def face_filter(image):
    img = np.float64(image) + 1.0
    intensity = np.sum(img, axis=2) / img.shape[2]
    retinex = multiScaleRetinex(intensity, [15, 101, 301])
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)
    intensity1 = simplestColorBalance(retinex, 0.01, 0.99)
    intensity1 = (intensity1 - np.min(intensity1)) / (np.max(intensity1) - np.min(intensity1)) * 255.0 + 1.0
    img_msrcp = np.zeros_like(img)
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]
        image = np.uint8(img_msrcp - 1.0)

    # cv2.imshow("img", image)

    #Median
    image = cv2.medianBlur(image, ksize=5)
    #Median
    image = cv2.medianBlur(image, ksize=5)
    #Canny
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(image, threshold1=30, threshold2=27)
    image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    #Threshold_binary_inverse
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

# threshold
def threshold(img, val, type):
    # 0 : Binary (cv2.THRESH_BINARY)
    # 1 : Binary Inverted
    # 2 : Truncate
    # 3 : To Zero
    # 4 : To Zero Inverted
    ret, threshold_img = cv2.threshold(img, val, 255, type)
    return threshold_img
# =========

# 사용자가 선택한 이진화 필터를 적용한 이미지를 반환해주는 함수
def binary_filter(id, image):

    if id == 1:
        # 1. Binarization > Threshold_binary 100
        # Threshold_binary
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 2:
        # 2. Binarization > Threshold_binary 150
        # Threshold_binary
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 3:
        # 3. Binarization > Threshold_otsu
        # Threshold_otsu
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 4:
        # 4. Adaptive_threshold_gaussian_c -> Threshold_binary_inverse
        # Adaptive_threshold_gaussian_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=3, C=0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 5:
        # 5. Adaptive_threshold_gaussian_c
        # Adaptive_threshold_gaussian_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=7, C=3)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 6:
        # 6. Adaptive_threshold_gaussian_c
        # Adaptive_threshold_gaussian_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=15, C=5)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 7:
        # 7. Adaptive_threshold_mean_c
        # Adaptive_threshold_mean_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=21, C=5)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 8:
        # 8. Original Canny - 50
        # Canny
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(image, threshold1=50, threshold2=50)
        image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 9:
        # 9. Original Canny - 100
        # Canny
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(image, threshold1=100, threshold2=100)
        image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 10:
        # 10. Edge Detection - Difference of gaussian
        # Difference_of_gaussian
        gaussian_1 = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=35, sigmaY=35)
        gaussian_2 = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=0)
        gaussian_1_gray = cv2.cvtColor(gaussian_1, cv2.COLOR_RGB2GRAY)
        gaussian_2_gray = cv2.cvtColor(gaussian_2, cv2.COLOR_RGB2GRAY)
        dog = gaussian_1_gray - gaussian_2_gray
        image = cv2.cvtColor(dog, cv2.COLOR_GRAY2RGB)


    elif id == 11:
        # 11. Edge Detection - Difference of gaussian
        # Difference_of_gaussian
        gaussian_1 = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=35, sigmaY=35)
        gaussian_2 = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=0)
        gaussian_1_gray = cv2.cvtColor(gaussian_1, cv2.COLOR_RGB2GRAY)
        gaussian_2_gray = cv2.cvtColor(gaussian_2, cv2.COLOR_RGB2GRAY)
        dog = gaussian_1_gray - gaussian_2_gray
        image = cv2.cvtColor(dog, cv2.COLOR_GRAY2RGB)

        # [추가]
        # Threshold_binary
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=109, maxval=255, type=cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


    elif id == 12:
        # 12. Edge Detection - Difference of gaussian -> Threshold_binary_inverse
        # Difference_of_gaussian
        gaussian_1 = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=0, sigmaY=194)
        gaussian_2 = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=255)
        gaussian_1_gray = cv2.cvtColor(gaussian_1, cv2.COLOR_RGB2GRAY)
        gaussian_2_gray = cv2.cvtColor(gaussian_2, cv2.COLOR_RGB2GRAY)
        dog = gaussian_1_gray - gaussian_2_gray
        image = cv2.cvtColor(dog, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=165, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 13:
        # 13. Emboss -> Threshold_binary
        # Emboss
        kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.filter2D(image, cv2.CV_8U, kernel_emboss)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=122, maxval=255, type=cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 14:
        # 14. Emboss -> Threshold_otsu
        # Emboss
        kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.filter2D(image, cv2.CV_8U, kernel_emboss)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_otsu
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 15:
        # 15. High_pass
        # High_pass
        kernel3x3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, cv2.CV_8U, kernel3x3)
        # Threshold_binary
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=31, maxval=255, type=cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 16:
        # 16. Prewitt
        # Prewitt
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        Kernel_X = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        Kernel_Y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        grad_x = cv2.filter2D(image, cv2.CV_16S, Kernel_X)
        grad_y = cv2.filter2D(image, cv2.CV_16S, Kernel_Y)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        image = cv2.addWeighted(abs_grad_x, 10 / 10, abs_grad_y, 11 / 10, 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=80, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 17:
        # 17. Prewitt
        # Prewitt
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        Kernel_X = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        Kernel_Y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        grad_x = cv2.filter2D(image, cv2.CV_16S, Kernel_X)
        grad_y = cv2.filter2D(image, cv2.CV_16S, Kernel_Y)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        image = cv2.addWeighted(abs_grad_x, 10 / 10, abs_grad_y, 11 / 10, 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_otsu
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=136, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 18:
        # 18. Roberts - 50
        # Roberts
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        Kernel_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        Kernel_Y = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
        grad_x = cv2.filter2D(image, cv2.CV_16S, Kernel_X)
        grad_y = cv2.filter2D(image, cv2.CV_16S, Kernel_Y)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        image = cv2.addWeighted(abs_grad_x, 30 / 10, abs_grad_y, 30 / 10, 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 19:
        # 19. Roberts - 100
        # Roberts
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        Kernel_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        Kernel_Y = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
        grad_x = cv2.filter2D(image, cv2.CV_16S, Kernel_X)
        grad_y = cv2.filter2D(image, cv2.CV_16S, Kernel_Y)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        image = cv2.addWeighted(abs_grad_x, 30 / 10, abs_grad_y, 30 / 10, 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 20:
        # 20. Scharr - 50
        # Scharr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scharrx = cv2.Scharr(image, ddepth=-1, dx=1, dy=0)
        scharry = cv2.Scharr(image, ddepth=-1, dx=0, dy=1)
        image = scharrx + scharry
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 21:
        # 21. Scharr - 100
        # Scharr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scharrx = cv2.Scharr(image, ddepth=-1, dx=1, dy=0)
        scharry = cv2.Scharr(image, ddepth=-1, dx=0, dy=1)
        image = scharrx + scharry
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 22:
        # 22. Scharr - 150
        # Scharr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        scharrx = cv2.Scharr(image, ddepth=-1, dx=1, dy=0)
        scharry = cv2.Scharr(image, ddepth=-1, dx=0, dy=1)
        image = scharrx + scharry
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=150, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 23:
        # 23. Sobel - 30
        # Sobel
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        SobelX = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5, scale=1 / 10)
        SobelY = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5, scale=1 / 10)
        sobel = abs(SobelX) + abs(SobelY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sobel)
        SobelImage = cv2.convertScaleAbs(sobel, alpha=-255 / max_val, beta=0)
        image = cv2.cvtColor(SobelImage, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=30, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 24:
        # 24. Sobel - 60
        # Sobel
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        SobelX = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5, scale=1 / 10)
        SobelY = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5, scale=1 / 10)
        sobel = abs(SobelX) + abs(SobelY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sobel)
        SobelImage = cv2.convertScaleAbs(sobel, alpha=-255 / max_val, beta=0)
        image = cv2.cvtColor(SobelImage, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=60, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 25:
        # 25. Sobel - 90
        # Sobel
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        SobelX = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5, scale=1 / 10)
        SobelY = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5, scale=1 / 10)
        sobel = abs(SobelX) + abs(SobelY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sobel)
        SobelImage = cv2.convertScaleAbs(sobel, alpha=-255 / max_val, beta=0)
        image = cv2.cvtColor(SobelImage, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=90, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 26:
        # 26. Dilation(밝은 부분 강조, 더 크게) -> Canny -> Threshold_binary_inverse
        # Dilation
        kernel = np.ones((6, 6), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        # Canny
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(image, threshold1=100, threshold2=0)
        image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=191, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 27:
        # 27. Erosion(어두운 부분 강조, 더 크게) -> Canny -> Threshold_binary_inverse
        # Erosion
        kernel = np.ones((6, 6), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        # Canny
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(image, threshold1=100, threshold2=0)
        image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        # Threshold_binary_inverse
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=191, maxval=255, type=cv2.THRESH_BINARY_INV)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 28:
        # 28. MSRCP -> Threshold_binary
        img = np.float64(image) + 1.0
        intensity = np.sum(img, axis=2) / img.shape[2]
        retinex = multiScaleRetinex(intensity, [15, 101, 301])
        intensity = np.expand_dims(intensity, 2)
        retinex = np.expand_dims(retinex, 2)
        intensity1 = simplestColorBalance(retinex, 0.01, 0.99)
        intensity1 = (intensity1 - np.min(intensity1)) / (np.max(intensity1) - np.min(intensity1)) * 255.0 + 1.0
        img_msrcp = np.zeros_like(img)
        for y in range(img_msrcp.shape[0]):
            for x in range(img_msrcp.shape[1]):
                B = np.max(img[y, x])
                A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
                img_msrcp[y, x, 0] = A * img[y, x, 0]
                img_msrcp[y, x, 1] = A * img[y, x, 1]
                img_msrcp[y, x, 2] = A * img[y, x, 2]
            image = np.uint8(img_msrcp - 1.0)
        # Threshold_binary
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, thresh=167, maxval=255, type=cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 29:
        # 만화필터
        # sigma_s : 이미지가 얼마나 스무스해질지를 결정, 클수록 더 스무스해진다. (0~200)
        # sigma_r : 이미지가 스무스해지는 동안 엣지를 얼마나 보존시킬지 결정. 작을 수록 엣지가 더 많이 보존된다. (0~1)
        image = cv2.stylization(image, sigma_s=40, sigma_r=0.05)
        image = grayscale(image)
        image = image.astype(np.uint8)
        image = threshold(image, 100, 0)

        # 원본 image이 shape이 달라지는 걸 방지하기 위해 추가한 코드
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif id == 30:
        # face 필터 (은진씨 코드)
        image = face_filter(image)

    elif id == 31:
        # feature 필터 (내가 만든 코드)
        image = feature_filter(image, 15, 200, 200)

        # 원본 image이 shape이 달라지는 걸 방지하기 위해 추가한 코드
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # [추가]
    elif id == 32:
        # 7. Adaptive_threshold_mean_c
        # Adaptive_threshold_mean_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=7, C=5)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # [추가]
    elif id == 33:
        # 7. Adaptive_threshold_mean_c
        # Adaptive_threshold_mean_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=13, C=5)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # [추가]
    elif id == 34:
        # 7. Adaptive_threshold_mean_c
        # Adaptive_threshold_mean_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=13, C=7)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # [추가]
    elif id == 35:
        # Difference_of_gaussian
        gaussian_1 = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=35, sigmaY=35)
        gaussian_2 = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=0)
        gaussian_1_gray = cv2.cvtColor(gaussian_1, cv2.COLOR_RGB2GRAY)
        gaussian_2_gray = cv2.cvtColor(gaussian_2, cv2.COLOR_RGB2GRAY)
        dog = gaussian_1_gray - gaussian_2_gray
        image = cv2.cvtColor(dog, cv2.COLOR_GRAY2RGB)
        # Adaptive_threshold_gaussian_c
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=3, C=8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image