# pix2pix, cycleGAN, colorization 알고리즘 학습에 필요한 dataset 만들기
# 32번 이진화 필터 사용

import cv2
import glob
import os

import binary_filterlist

# 데이터셋을 저장할 기본 디렉토리
# 저장 시 사용할 경로
# base_dir = "D:/MARS/exp/exp44/dataset/"
# base_dir = 'D:/210106/pix_asian1_dataset/'
base_dir = 'C:/Users/jaehee/Documents/210106/pix_asian2_dataset/'

# 데이터 불러올 때 사용할 경로 'D:/MARS/exp/exp44/test/*/'
# dataset_dir = "D:/MARS/exp/exp44/test/*/"
# dataset_dir = 'D:/210106/pix_asian1/*/'
dataset_dir = 'C:/Users/jaehee/Documents/210106/pix_asian2/*/'

# ============
# << 디렉토리에 필요한 폴더가 없으면 생성하는 함수 >>
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            # print(directory + ' 생성 완료!')
    except OSError:
        print('Error: Creating directory. ' + directory)

# << pix2pix 채색에 필요한 형태로 데이터셋을 만드는 함수 >>
# obj_nm : 객체명(=폴더명), datatype : train/test/val, img_nm : 이미지명, img : 원본 이미지
# filter_id : 필터id, b_img : 이진화 이미지
# train, val, test  // 이진화 이미지 + 컬러 이미지
def save_pix2pix(obj_nm, datatype, img_nm, img, filter_id, b_img):

    # 데이터셋 폴더명(객체명_채색타입) (ex) avocados_color
    dataset_nm = obj_nm + '_pix_' + filter_id

    # 데이터셋 폴더 생성
    createFolder(base_dir + dataset_nm)
    # test, train, val 데이터 폴더 생성
    createFolder(base_dir + dataset_nm + '/' + datatype)
    save_path = base_dir + dataset_nm + '/' + datatype + '/'

    merged_img = cv2.hconcat([b_img, img])

    # 저장
    cv2.imwrite(save_path + img_nm + '.jpg', merged_img)

# << pix2pix 채색에 필요한 형태로 데이터셋을 만드는 함수(컬러 -> 흑백) >>
# obj_nm : 객체명(=폴더명), datatype : train/test/val, img_nm : 이미지명, img : 원본 이미지
# filter_id : 필터id, b_img : 이진화 이미지
# train, val, test  // 이진화 이미지 + 컬러 이미지
def save_pix2pix_rev(obj_nm, datatype, img_nm, img, filter_id, b_img):

    # 데이터셋 폴더명(객체명_채색타입) (ex) avocados_color
    dataset_nm = obj_nm + '_pix_' + filter_id

    # 데이터셋 폴더 생성
    createFolder(base_dir + dataset_nm)
    # test, train, val 데이터 폴더 생성
    createFolder(base_dir + dataset_nm + '/' + datatype)
    save_path = base_dir + dataset_nm + '/' + datatype + '/'

    merged_img = cv2.hconcat([img, b_img])

    # 저장
    cv2.imwrite(save_path + img_nm + '.jpg', merged_img)

# << cycleGAN 채색에 필요한 형태로 데이터셋을 만드는 함수 >>
# obj_nm : 객체명(=폴더명), datatype : train/test/val, img_nm : 이미지명, img : 원본 이미지
# filter_id : 필터id, b_img : 이진화 이미지
# trainA, trainB // 이진화이미지, 컬러이미지
def save_cyclegan(obj_nm, datatype, img_nm, img, filter_id, b_img):

    # 데이터셋 폴더명(객체명_채색타입_필터id) (ex) avocados_cycle_1
    dataset_nm = obj_nm + '_cycle_' + filter_id

    # 데이터셋 폴더 생성
    createFolder(base_dir + dataset_nm)
    # test, train, val 데이터 폴더 생성 (trainA, trainB)
    createFolder(base_dir + dataset_nm + '/' + datatype + 'A')
    createFolder(base_dir + dataset_nm + '/' + datatype + 'B')

    save_path1 = base_dir + dataset_nm + '/' + datatype + 'A/'
    save_path2 = base_dir + dataset_nm + '/' + datatype + 'B/'

    # 저장
    cv2.imwrite(save_path1 + img_nm + '.jpg', b_img)
    cv2.imwrite(save_path2 + img_nm + '.jpg', img)


# << colorization 채색에 필요한 형태로 데이터셋을 만드는 함수 >>
# train, test, val  // 컬러 이미지
def save_color(obj_nm, datatype, img_nm, img):

    # 데이터셋 폴더명(객체명_채색타입) (ex) avocados_color
    dataset_nm = obj_nm + '_color'

    # 데이터셋 폴더 생성
    createFolder(base_dir + dataset_nm)
    # test, train, val 데이터 폴더 생성
    createFolder(base_dir + dataset_nm + '/' + datatype)
    save_path = base_dir + dataset_nm + '/' + datatype + '/'

    # 저장
    cv2.imwrite(save_path + img_nm + '.jpg', img)
# ============

# main 함수
# << data 디렉토리에 있는 전체 객체 폴더 돌면서, 필요한 이미지 형태로 데이터셋 만들기 >>
# [우분투 변경 필요] data 디렉토리에 있는 전체 객체폴더 경로 가져오기
folder_list = glob.glob(dataset_dir)
# print(folder_list)

# 객체 폴더 하나씩 돌면서 데이터셋 만들기
for fname in folder_list:

    # [우분투 변경 필요] 폴더명 = 객체명
    folder_nm = fname.split('\\')[1]

    # train, test, val 폴더의 이미지 모두 가져오기
    train_imglist = glob.glob(fname + '/train/*.jpg')
    test_imglist = glob.glob(fname + '/test/female_long/*.jpg')
    val_imglist = glob.glob(fname + '/val/*.jpg')

    # Q. 이미지명을 원래 값을 유지해야 할 이유가 있을까? 새로운 id 값으로 발급하면?
    img_nm = 901

    # << Train 데이터셋 저장 >>
    # 이미지 개수만큼 반복하면서 필터, 채색 알고리즘 별 데이터셋 생성
    for iname in train_imglist:
        image = cv2.imread(iname)

        # [추가] 그레이스케일 이미지
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        # 채색 타입에 맞게 데이터셋 저장 - colorization (이진화 이미지 필요 없음)
        # save_color(folder_nm, 'train', str(img_nm), image)

        # 필터 개수만큼 반복하기(1~31번 필터) - pix2pix, cycleGAN
        for i in range(32, 33):
            b_image = binary_filterlist.binary_filter(i, image)

            # 채색 타입에 맞게 데이터셋 저장
            # pix2pix
            # save_pix2pix(folder_nm, 'train', str(img_nm), image, str(i), b_image)
            save_pix2pix(folder_nm, 'train', str(img_nm), image, str(i), b_image)
            # save_pix2pix_rev(folder_nm, 'train', str(img_nm), image, str(i), b_image)

            # cyclegan
            # save_cyclegan(folder_nm, 'train', str(img_nm), image, str(i), b_image)

        print("[train] " + folder_nm + " / " + str(img_nm) + " 저장 완료!")

        img_nm = img_nm + 1


    # << Test 데이터셋 저장 >>
    # 이미지 개수만큼 반복하면서 필터, 채색 알고리즘 별 데이터셋 생성
    # for iname in test_imglist:
    #     image = cv2.imread(iname)
    #
    #     # [추가] 그레이스케일 이미지
    #     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    #
    #     # 채색 타입에 맞게 데이터셋 저장 - colorization (이진화 이미지 필요 없음)
    #     # save_color(folder_nm, 'test', str(img_nm), image)
    #
    #     # 필터 개수만큼 반복하기 - pix2pix, cycleGAN
    #     for i in range(32, 33):
    #         b_image = binary_filterlist.binary_filter(i, image)
    #
    #         # 채색 타입에 맞게 데이터셋 저장
    #         # pix2pix
    #         # save_pix2pix(folder_nm, 'test', str(img_nm), image, str(i), b_image)
    #         # save_pix2pix_rev(folder_nm, 'test', str(img_nm), image, str(i), b_image)
    #         save_pix2pix(folder_nm, 'test', str(img_nm), image, str(i), b_image)
    #         # cyclegan
    #         # save_cyclegan(folder_nm, 'test', str(img_nm), image, str(i), b_image)
    #
    #     print("[test] " + folder_nm + " / " + str(img_nm) + " 저장 완료!")
    #
    #     img_nm = img_nm + 1


    # << Val 데이터셋 저장 >>
    # 이미지 개수만큼 반복하면서 필터, 채색 알고리즘 별 데이터셋 생성
    # for iname in val_imglist:
    #     image = cv2.imread(iname)
    #
    #     # [추가] 그레이스케일 이미지
    #     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    #
    #     # 채색 타입에 맞게 데이터셋 저장 - colorization (이진화 이미지 필요 없음)
    #     save_color(folder_nm, 'val', str(img_nm), image)
    #
    #     # 필터 개수만큼 반복하기 - pix2pix, cycleGAN
    #     for i in range(32, 33):
    #         b_image = binary_filterlist.binary_filter(i, image)
    #
    #         # 채색 타입에 맞게 데이터셋 저장
    #         # pix2pix
    #         # save_pix2pix(folder_nm, 'val', str(img_nm), image, str(i), b_image)
    #         # save_pix2pix_rev(folder_nm, 'val', str(img_nm), image, str(i), b_image)
    #         save_pix2pix(folder_nm, 'val', str(img_nm), image, str(i), b_image)
    #         # cyclegan
    #         # save_cyclegan(folder_nm, 'val', str(img_nm), image, str(i), b_image)
    #
    #     print("[val] " + folder_nm + " / " + str(img_nm) + " 저장 완료!")
    #
    #     img_nm = img_nm + 1


    print("<< " + folder_nm + " 저장완료 >>")





