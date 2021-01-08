# 작성자  : 배재희
# 설  명  : 자주 사용하는 기능들을 모아 둔 코드

import cv2
from PIL import Image
import numpy as np

# << openCV 이미지를 PIL Image로 바꾸는 함수 >>
def cvtopil(cvimg):
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cvimg)
    return pil_img

# << PIL Image를 openCV 이미지로 바꾸는 함수 >>
def piltocv(pilimg):
    # PIL 이미지를 numpy 배열로 변환(openCV는 이미지를 numpy array로 저장)
    numpy_image = np.array(pilimg)
    # openCV 타입의 이미지의 기본 색상포맷인 BGR로 변경 (변경하지 않으면, 파란 색상의 이미지가 나옴)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image

# << 찾고자하는 색깔로 영역을 찾은 후 투명한 픽셀로 변경해주는 함수 >>
# openCV 이미지는 배경색이 투명한 png 이미지 지원이 안되기 때문에 PIL 이미지로 변경하여 사용
def make_transparent_img(mask_img, r=0, g=0, b=0):

    # openCV 이미지를 PIL 이미지로 변경
    # 나중에 PIL 이미지가 입력되더라도 오류나지 않도록 이미지 타입 체크해서 opencv일 때만 변경하도록 수정할 것
    pil_img = cvtopil(mask_img)

    # 투명도를 포함한 RGBA 타입으로 변경
    pil_img = pil_img.convert("RGBA")

    # 픽셀 단위 색상 정보(RGBA)를 가지고 있는 datas 변수
    datas = pil_img.getdata()
    # 변경된 최종 픽셀 정보를 담기 위한 변수
    newData = []

    # PIL 이미지에서 마스크(검은색)된 픽셀만 찾아서 투명 픽셀로 변경
    for item in datas:
        if item[0] == r and item[1] == g and item[2] == b:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    pil_img.putdata(newData)

    return pil_img

# 찾고자하는 색깔(r, g, b 값)로 영역을 찾은 후 영역을 검은색으로 변경
def make_black_img(mask_img, r=0, g=0, b=0):

    # openCV 이미지를 PIL 이미지로 변경
    pil_img = cvtopil(mask_img)

    # 투명도를 포함한 RGBA 타입으로 변경
    pil_img = pil_img.convert("RGBA")

    # 픽셀 단위 색상 정보(RGBA)를 가지고 있는 datas 변수
    datas = pil_img.getdata()
    # 변경된 최종 픽셀 정보를 담기 위한 변수
    newData = []

    # PIL 이미지에서 마스크(검은색)된 픽셀만 찾아서 투명 픽셀로 변경
    for item in datas:
        if item[0] == r and item[1] == g and item[2] == b:
            newData.append((0, 0, 0))
        else:
            newData.append((255, 255, 255, 0))
            # newData.append(item)
    pil_img.putdata(newData)

    return pil_img

# 남기고 싶은 색깔(r, g, b 값)만 빼고 다 투명으로 만들어주는 함수
def make_transparent_leftcolor(img, r=0, g=0, b=0):
    # openCV 이미지를 PIL 이미지로 변경
    pil_img = cvtopil(img)

    # 투명도를 포함한 RGBA 타입으로 변경
    pil_img = pil_img.convert("RGBA")

    # 픽셀 단위 색상 정보(RGBA)를 가지고 있는 datas 변수
    datas = pil_img.getdata()
    # 변경된 최종 픽셀 정보를 담기 위한 변수
    newData = []

    # PIL 이미지에서 마스크(검은색)된 픽셀만 찾아서 투명 픽셀로 변경
    for item in datas:
        if item[0] == r and item[1] == g and item[2] == b:
            newData.append(item)
            # newData.append((0, 0, 0))
        else:
            newData.append((255, 255, 255, 0))
            # newData.append(item)
    pil_img.putdata(newData)

    return pil_img

# [추가] 입력받은 색은 투명으로 바꾸고 나머지는 다 검정색으로 만들어주는 함수
def make_black_leftcolor(img, r=255, g=255, b=255):
    # openCV 이미지를 PIL 이미지로 변경
    pil_img = cvtopil(img)

    # 투명도를 포함한 RGBA 타입으로 변경
    pil_img = pil_img.convert("RGBA")

    # 픽셀 단위 색상 정보(RGBA)를 가지고 있는 datas 변수
    datas = pil_img.getdata()
    # 변경된 최종 픽셀 정보를 담기 위한 변수
    newData = []

    # PIL 이미지에서 마스크(검은색)된 픽셀만 찾아서 투명 픽셀로 변경
    for item in datas:
        if item[0] == r and item[1] == g and item[2] == b:
            newData.append((255, 255, 255, 0))
        else:
            newData.append((0, 0, 0))
    pil_img.putdata(newData)

    return pil_img

# << 입력받은 이미지를 흰색으로 만들어 주는 함수 >>
def make_white_img(trg_img, is_pil=True):

    img = trg_img.copy()

    if is_pil == False:
        img = cvtopil(img)
        # 투명도를 포함한 RGBA 타입으로 변경
        img = img.convert("RGBA")

    # 픽셀 단위 색상 정보(RGBA)를 가지고 있는 datas 변수
    datas = img.getdata()
    # 변경된 최종 픽셀 정보를 담기 위한 변수
    newData = []

    # PIL 이미지의 모든 픽셀의 색을 흰색으로 변경
    for item in datas:
        if is_pil==True:
            if item[0] == 255 and item[1] == 255 and item[2] == 255 and item[3] == 0:
                # 투명색인 픽셀
                newData.append(item)
                # print("투명색!")
            else :
                # 투명색 아닌 픽셀은 흰색으로 변경
                newData.append((255, 255, 255))
        else:
            newData.append((255,255,255))
    img.putdata(newData)

    if is_pil == False:
        img = piltocv(img)
        # print("white img shape : ", img.shape)

    return img

# << 컨투어를 이용해 원하는 영역 지정 후 필요한 부분만 남기는 함수 >>
# mask_and_crop_hull로 변경할 것
def mask_and_crop(img, hull):
    # 검정색 마스크 이미지
    mask_img = np.zeros(img.shape[0:2], dtype=np.uint8) * 255

    # 형태에 맞춰 자를 부위 영역만 흰색으로 내부 채우기
    cv2.drawContours(mask_img, [hull], -1, (255, 255, 255), -1)

    # 원본 이미지와 비트연산하여 부위 영역만 남기기 (나머지 영역은 검은색)
    left_img = cv2.bitwise_and(img, img, mask=mask_img)

    # 부위 형태 크기에 맞게 바운딩 박스 잡기
    rect = cv2.boundingRect(hull)

    # 바운딩 박스 형태로 자르기
    crop_img = left_img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    # 투명한 이미지로 변경
    crop_img = make_transparent_img(crop_img)

    return crop_img

# << 남길 픽셀의 x, y 좌표 값을 이용해 필요한 부분의 형태만 남기는 함수 >>
def mask_and_crop_xy(img, x, y, x_padding, y_padding):
    mask_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # 마스크 이미지에서 남길 영역의 픽셀만 원본 이미지 픽셀로 붙여넣기
    mask_img[y, x, :] = img[y, x, :]

    # crop the eye_mask image
    # (1) find x,y min / max pixel
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    start_x = min_x - x_padding
    start_y = min_y - y_padding
    last_x = min_x + (max_x - min_x) + x_padding
    last_y = min_y + (max_y - min_y) + y_padding

    # exception
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0

    if last_x < 0:
        last_x = 0
    if last_x > mask_img.shape[1]:
        last_x = mask_img.shape[1]

    if last_y < 0:
        last_y = 0
    if last_y > mask_img.shape[0]:
        last_y = mask_img.shape[0]

    # (2) crop using bounding box
    cropped_img = mask_img[start_y:last_y, start_x:last_x]

    cropped_pilimg = make_transparent_img(cropped_img)

    return cropped_pilimg

# << 컨투어를 이용해 원하는 영역에 라인을 그리는 함수 - hull 이용 >>
# img : 컨투어 라인을 그릴 이미지
# hull : 컨투어 영역의 좌표를 저장하고 있는 변수
def draw_contourline_hull(img, hull):

    # 윤곽선 내부에 흰색 채우기
    cv2.drawContours(img, [hull], -1, (255, 255, 255), -1)

    # 원본 이미지에 검정색으로 라인 그리기
    cv2.drawContours(img, [hull], -1, (0, 0, 0), 1)
    # cv2.imshow("img", img)
    return img

def find_and_draw_contourline(mask_img, img):

    # cv2.imshow("mask_img", mask_img)
    # cv2.imshow("img", img)

    # img가 마스크된 이미지이면 그대로 사용하고
    # 마스크된 이미지가 아니면 변경 필요

    # findContours 윤곽선을 찾아주는 함수
    # cv2.findContours(image, mode, method)
    # mode : contours를 찾는 방법
    # cv2.RETR_EXTERNAL -> contours line 중 가장 바깥쪽 라인만 찾음
    # cv2.RETR_LIST -> 모든 contours line을 찾지만, 계층 관계를 구성하지 않음
    # method : contours를 찾을 때 사용하는 근사치 방법
    # cv2.CHAIN_APPROX_NONE -> 모든 contours point를 저장
    # cv2.CHAIN_APPROX_SIMPLE -> contours line을 그릴 수 있는 point만 저장
    contours, _ = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 라인 그리기
    # 원본 이미지에 경계선 그리기 (모든 컨투어 그리기) -> 눈썹, 눈, 코, 입 모두 그림 (skin이 들어왔을 때 모든 컨투어를 그려서)
    # cv2.drawContours(image, contours, contouridx, color, thickness)
    # print("drawContours!")
    # cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

    for cnt in contours:
        # 윤곽선 내부에 흰색 채우기
        cv2.drawContours(img, [cnt], -1, (255, 255, 255), -1)

        # 윤곽선 그리기 (0번째 컨투어만 그리기)
        cv2.drawContours(img, [cnt], 0, (0, 0, 0), 2)

    return img


# 원본 이미지와 좌표값을 전달 받아서 네모로 자르는 함수
def crop_rec_img(img, start_x, last_x, start_y, last_y, x_padding, y_padding):
    crop_img = img[start_y-y_padding : last_y+y_padding, start_x-x_padding:last_x+x_padding]
    return crop_img
