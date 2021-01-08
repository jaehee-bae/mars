# 얼굴 데이터셋에 이진화 필터를 적용하는 코드(1~35가지 이진화 필터)
# 원본 이미지에 바이너리 필터 적용 후, 결과 이미지를 이어붙여서 비교하기 쉽게 저장하는 코드

import glob
import cv2
from PIL import Image
import os

import binary_filterlist
from etc import utils


# 디렉토리에 폴더가 없으면 생성하는 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory + ' 생성 완료!')
    except OSError:
        print('Error: Creating directory. ' + directory)

# 이미지 병합 함수
# x_size는 가로 총 길이가 와야함
def imageMerge(file_list, x_size, x_min, y_min):

    print(x_size, x_min, y_min)

    # 하얀 도화지의 크기 / 가로 8, 세로 4
    new_image = Image.new("RGB", (x_size, y_min), (256,256,256))
    # new_image.open()

    for index in range(len(file_list)):
        # area : 이미지를 병합해줄 때 서로 다른 이미지를 겹쳐주게 할 위치값이 됨
        # 시작점 가로, 시작점 세로, 이미지 가로크기, 이미지 세로크기
        # (0, 0, 256, 256), (256, 0, 512, 256), (512, 0, 256*3, 256), index:3 (256*3, 0, 256*4, 256)
        area = ((index * x_min), 0, (x_min * (index+1)), y_min)
        new_image.paste(file_list[index], area)
    return new_image

# ==== main ====
# 전체 객체 폴더 리스트
# folder_list = glob.glob('D:/MARS/exp/exp40/dataset/fruit/*/')
folder_list = glob.glob('D:/pix_asian1/add1/*/')
# print(folder_list)

# 객체 폴더 전체 돌면서 실행
for fname in folder_list:
    folder_nm = fname.split('\\')[1]
    print(fname)

    # 객체 폴더 안에 있는 모든 이미지 불러오기
    # train, test, val 나눠져있는 경우
    # img_list = glob.glob(fname + '/*/*.jpg')
    # 나눠져있지 않은 경우
    img_list = glob.glob(fname + '/*.jpg')

    # print(img_path)

    # 이미지 id값 1로 세팅
    img_id = 1

    # 이미지 리스트
    print(img_list)

    # 이미지 불러오기
    for iname in img_list:

        image = cv2.imread(iname)

        # 이미지 사이즈 변경
        # 특정한 이미지에 대해서 오류 발생(8비트 이미지일 경우 오류 발생, 화질 자체가 떨어져서 경로에서 이미지 삭제함)
        # cv2.error: OpenCV(4.4.0) C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-2b5g8ysb\opencv\modules\imgproc\src\resize.cpp:3929: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'
        # image = cv2.resize(image, (256, 256))

        # 폴더명이 없으면 폴더 생성 (folder_nm 활용)
        createFolder('D:/pix_asian_b1/' + folder_nm)

        # 병합할 이미지를 담을 파일 리스트 선언
        file_list = []

        # 바이너리 이미지로 만들기
        # i=0 원본 이미지, i=1~35 이진화 이미지
        for i in range(0, 36):

            print("filter num : ", i)

            b_image = image.copy()
            b_image = cv2.resize(b_image, (256, 256))

            if i == 0:
                # 원본일 경우
                # print("original image")
                b_image = utils.cvtopil(b_image)

                # save_img = image.copy()
                # save_img = cv2.resize(save_img, (512, 512))

                # 저장
                # 이미지 저장 (data/avocados/avocados_1.jpg)
                # save_path = 'D:/MARS/exp/exp40/analyze/' + folder_nm + '/' + folder_nm + '_' + str(
                #     img_id) + '.jpg'
                # cv2.imwrite(save_path, save_img)

            else :
                # 이진화 필터 적용
                b_image = binary_filterlist.binary_filter(i, b_image)

                b_image = utils.piltocv(b_image)
                cv2.putText(b_image, str(i), (0,253), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), thickness=3)
                b_image = utils.cvtopil(b_image)


            # 병합할 리스트에 담기
            file_list.append(b_image)


        # 이미지 비교를 위해 연한 것 ~ 진한 것 순서대로 담기
        m1_list = []
        m2_list = []
        m3_list = []
        m4_list = []
        m5_list = []
        m6_list = []

        # 2. 연한 것부터 진한 순서대로 이미지 배열하기 (편하게 이미지 비교하기 위함)
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

        # 이미지 저장 (data/avocados/avocados_1.jpg)
        save_path = 'D:/pix_asian_b1/' + folder_nm + '/' + folder_nm + '_' + str(img_id) + '_merged.jpg'
        cv2.imwrite(save_path, final_img)

        # 이미지 id 값 증가
        img_id = img_id + 1

    print('<<' + folder_nm + ' 객체 저장 완료! >>')