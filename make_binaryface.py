# 원본 이미지에 35가지 이진화 필터 적용 후, 결과 이미지를 이어붙여서 비교하기 쉽게 저장하는 코드

import glob
import cv2
from PIL import Image

import binary_filterlist
import utils

# 이미지 병합 함수
# x_size :가로 총 길이
def imageMerge(file_list, x_size, x_min, y_min):

    # print(x_size, x_min, y_min)

    # 하얀 도화지의 크기 / 가로 8, 세로 4
    new_image = Image.new("RGB", (x_size, y_min), (256,256,256))
    # new_image.open()

    for index in range(len(file_list)):
        # area : 이미지를 병합해줄 때 서로 다른 이미지를 겹쳐주게 할 위치값이 됨
        # 시작점 가로, 시작점 세로, 이미지 가로크기, 이미지 세로크기
        area = ((index * x_min), 0, (x_min * (index+1)), y_min)
        new_image.paste(file_list[index], area)
    return new_image

# << main >>
if __name__ == "__main__":

    # 원본 얼굴 이미지가 존재하는 경로
    IMG_DIR = 'data/face/*.jpg'
    # 최종 이미지를 저장할 경로
    SAVE_DIR = 'data/binary35/'

    images = glob.glob(IMG_DIR)
    # print(images)

    # 이미지 파일명을 얻기위해 필요
    # 우분투일 경우, '/'
    # 윈도우일 경우, '\\'
    slush_count = images[0].count('\\')
    # print(slush_count)

    # 이미지 불러오기
    for fname in images:

        # 저장 시 사용하기 위한 원본 얼굴 이미지 파일명 (여러 장일 경우)
        # 우분투일 경우, '/'
        # 윈도우일 경우, '\\'
        img_name = fname.split('\\')[slush_count].split('.')[0]
        # print(img_name)

        # 원본 얼굴 이미지
        image = cv2.imread(fname)

        # 병합할 이미지를 담을 파일 리스트 선언
        file_list = []

        # 바이너리 이미지로 만들기
        # i=0 원본 이미지, i=1~35 이진화 이미지
        for i in range(0, 36):

            # print("filter num : ", i)

            b_image = image.copy()
            b_image = cv2.resize(b_image, (256, 256))

            if i == 0:
                # 원본일 경우
                b_image = utils.cvtopil(b_image)
            else :
                # 이진화 필터 적용
                b_image = binary_filterlist.binary_filter(i, b_image)

                b_image = utils.piltocv(b_image)
                # 좌측 아래에 바이너리 필터 번호 텍스트 입력하기
                cv2.putText(b_image, str(i), (0,253), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=3)
                b_image = utils.cvtopil(b_image)

            # 병합할 리스트에 담기
            file_list.append(b_image)

        m1_list = []
        m2_list = []
        m3_list = []
        m4_list = []
        m5_list = []
        m6_list = []

        # 이미지 비교를 위해 연한 것 ~ 진한 것 순서대로 담기
        sort_map = [0, 15, 9, 27, 26, 8,
                    30, 19, 31, 25, 12, 24,
                    5, 17, 16, 6, 22, 21,
                    18, 23, 4, 20, 10, 7,
                    11, 2, 3, 1, 28, 13,
                    14, 29, 32, 33, 34, 35]

        width_num = 6

        for index in sort_map:
            if index in (0, 15, 9, 27, 26, 8):  # 가장 연한 이미지 (0은 원본이미지로 비교하기 편하게 가장 앞에 배치)
                m1_list.append(file_list[index])
                if len(m1_list) == width_num:
                    # 가로로 먼저 합치기 (6개)
                    m1 = imageMerge(m1_list, 256 * width_num, 256, 256)

            elif index in (30, 19, 31, 25, 12, 24):
                m2_list.append(file_list[index])
                if len(m2_list) == width_num:
                    # 가로로 먼저 합치기 (6개)
                    m2 = imageMerge(m2_list, 256 * width_num, 256, 256)

            elif index in (5, 17, 16, 6, 22, 21):
                m3_list.append(file_list[index])
                if len(m3_list) == width_num:
                    # 가로로 먼저 합치기
                    m3 = imageMerge(m3_list, 256 * width_num, 256, 256)

            elif index in (18, 23, 4, 20, 10, 7):
                m4_list.append(file_list[index])
                if len(m4_list) == width_num:
                    # 가로로 먼저 합치기
                    m4 = imageMerge(m4_list, 256 * width_num, 256, 256)

            elif index in (11, 2, 3, 1, 28, 13):
                m5_list.append(file_list[index])
                if len(m5_list) == width_num:
                    # 가로로 먼저 합치기
                    m5 = imageMerge(m5_list, 256 * width_num, 256, 256)

            elif index in (14, 29, 32, 33, 34, 35):
                m6_list.append(file_list[index])
                if len(m6_list) == width_num:
                    # 가로로 먼저 합치기
                    m6 = imageMerge(m6_list, 256 * width_num, 256, 256)

        # PIL Image -> OpenCV Image
        cv_m1 = utils.piltocv(m1)
        cv_m2 = utils.piltocv(m2)
        cv_m3 = utils.piltocv(m3)
        cv_m4 = utils.piltocv(m4)
        cv_m5 = utils.piltocv(m5)
        cv_m6 = utils.piltocv(m6)

        # 세로로 이미지 병합
        final_img = cv2.vconcat([cv_m1, cv_m2, cv_m3, cv_m4, cv_m5, cv_m6])

        # 이미지 저장
        cv2.imwrite(SAVE_DIR + img_name + "_merged.jpg", final_img)
