# 얼굴부위 조합을 위해 필요한 함수들을 모아놓은 코드

import cv2
import dlib
import glob
import random
import utils as util
import numpy as np


# << 얼굴 부위(눈, 눈썹, 코, 입) 이미지를 불러오는 함수 >>
# fname : 불러올 부위명 (오른쪽 눈:eye/코:nose/입:mouth/오른쪽 눈썹:eyebrow)
def load_face_features(fname):
    # 샘플이미지 경로
    file_path = 'sample_img/features/502/'
    # 생성한 부위 불러오기
    if fname == 'eye':
        # 오른쪽 눈
        file_path = file_path + 'reye/*.jpg'
        # file_path = file_path + 'eye/*.jpg'
    elif fname == 'nose':
        file_path = file_path + 'nose/*.jpg'
    elif fname == 'mouth':
        file_path = file_path + 'mouth/*.jpg'
    elif fname == 'eyebrow':
        file_path = file_path + 'reyebrow/*.jpg'
        # file_path = file_path + 'eyebrow/*.jpg'

    images = glob.glob(file_path)
    if len(images) == 0:
        print("Error! 이미지를 불러올 수 없습니다. 경로를 다시 확인해주세요.")
        return

    # 불러온 이미지 중에서 랜덤으로 1개 이미지 선택
    random_idx = random.randint(0, len(images) - 1)
    # print(random_idx)
    fname = images[random_idx]
    # print(fname)

    # 선택된 이미지 가져오기
    img = cv2.imread(fname)

    # 이미지에서 검정색 테두리 삭제 (BEGAN으로 생성한 이미지의 경우 검은색 테두리 생김)
    # img = img[2:img.shape[0] - 2, 2:img.shape[1] - 2]

    return img


# [추가]
# << 얼굴 부위(오른쪽눈, 왼쪽눈, 오른쪽눈썹, 왼쪽눈썹, 코, 입) 이미지를 불러오는 함수 >>
# fname : 불러올 부위명 (오른쪽 눈:eye/코:nose/입:mouth/오른쪽 눈썹:eyebrow)
# 선택된 이미지 번호 같이 리턴하도록 수정
def load_face_features_detail(file_path, fname):
    # 샘플이미지 경로
    # file_path = 'assemble_test1_features/'

    # 생성한 부위 불러오기
    if fname == 'eye':
        # 오른쪽 눈
        f_path = file_path + 'reye/*.jpg'
    elif fname == 'nose':
        f_path = file_path + 'nose/*.jpg'
    elif fname == 'mouth':
        f_path = file_path + 'mouth/*.jpg'
    elif fname == 'eyebrow':
        f_path = file_path + 'reyebrow/*.jpg'

    # print(fname)

    images = glob.glob(f_path)
    if len(images) == 0:
        print("Error! 이미지를 불러올 수 없습니다. 경로를 다시 확인해주세요.")
        return

    # 불러온 이미지 중에서 랜덤으로 1개 이미지 선택
    random_idx = random.randint(0, len(images) - 1)
    # print(random_idx)
    img_path = images[random_idx]
    # print(fname)

    # 선택된 이미지 가져오기
    img = cv2.imread(img_path)
    img2 = img.copy()

    # 이미지명 (ex) 3_leye
    img_name1 = img_path.split('/')[img_path.count('/')].split('.')[0]
    # print("img_name1 : ", img_name1)
    img_name2 = ''

    # 눈, 눈썹의 경우 쌍을 이루는 왼쪽 눈, 눈썹도 가져옴
    if fname in ('eye', 'eyebrow'):
        img_id = img_name1.split('_')[0]
        img_name2 = img_id + '_l' + fname
        # print("img_name2 : " + img_name2)
        img_path2 = file_path + 'l' + fname + '/' + img_name2 + '.jpg'
        # print("img_path2 : " + img_path2)

        # 왼쪽 눈, 눈썹 이미지 가져오기
        img2 = cv2.imread(img_path2)

        # 왼쪽 부위 이미지가 존재하지 않을 경우 오른쪽 부위 이미지를 가로로 뒤집어서 사용
        if img2 is None:
            # print("img2 is None")
            img2 = cv2.flip(img, 1)
            img_name2 = img_name1 + '_rev'

    # 이미지에서 검정색 테두리 삭제 (BEGAN으로 생성한 이미지의 경우 검은색 테두리 생김)
    # img = img[2:img.shape[0] - 2, 2:img.shape[1] - 2]

    return img, img2, img_name1, img_name2


# << 원본 얼굴 이미지(src_img) 위에 부위 이미지(trg_img)를 붙이는 함수 >>
# src_img : 원본 얼굴 이미지 / trg_img : 생성한 부위 이미지
# start_x, start_y : 부위이미지가 들어갈 원본 이미지 x, y 좌표 시작점
# last_x, last_y : 부위이미지가 들어갈 원본 이미지 x, y 좌표 끝점
# x_padding, y_padding : x, y 패딩 값. x, y 좌표 시작점과 끝점 값에 반영한다.
# is_PIL : True면 src_img, trg_img가 PIL 타입 이미지이고 False이면 openCV 이미지이다.
def attach_features(src_img, trg_img, start_x, last_x, start_y, last_y, x_padding, y_padding, is_PIL=False):

    # 패딩 값 적용하여 좌표값 찾기
    start_x = start_x - x_padding
    start_y = start_y - y_padding
    last_x = last_x + x_padding
    last_y = last_y + y_padding

    if is_PIL == True:
        # print("attach_features!")

        # 부위 별 이미지 resize((가로길이, 세로길이))
        trg_img = trg_img.resize((last_x - start_x, last_y - start_y))
        # print("resize features / width : ", last_x-start_x, " height : ", last_y-start_y)

        # PIL 이미지 붙여넣기
        src_img.paste(trg_img, (start_x, start_y), trg_img)
    else:

        # 부위 별 이미지 resize(이미지, (가로길이, 세로길이))
        trg_img = cv2.resize(trg_img, (last_x - start_x, last_y - start_y))

        # print("trg_img shape : ", trg_img.shape)
        # print("src_img shape : ", src_img.shape)

        # openCV 이미지 붙여넣기
        src_img[start_y:last_y, start_x:last_x] = trg_img

    return src_img


# << 부위 이미지를 조합하여 새로운 얼굴을 만드는 함수 (왼쪽 눈, 눈썹 추가) >>
# trg_righteye : 오른쪽 눈 이미지 / trg_righteyebrow : 오른쪽 눈썹 이미지
# trg_nose : 코 이미지 / trg_mouth : 입 이미지
# src_img : 원본 얼굴 이미지(랜드마크 활용하여 얼굴 검출 시 사용)
# is_PIL : True  - 인자로 전달받은 모든 이미지가 PIL 타입 이미지, 출력도 PIL 이미지 (세그멘테이션된 부위 이미지 조합 시 사용함)
#          False - 인자로 전달받은 모든 이미지가 openCV 이미지, 출력도 openCV 이미지
# trg_img : 부위 이미지를 붙여넣을 타겟 이미지
def assemble_features_detail(trg_righteye='None', trg_righteyebrow='None', trg_nose='None', trg_mouth='None', src_img='None', is_PIL=False, is_seg=False, trg_img='None',
                             trg_lefteye='None', trg_lefteyebrow='None', img_size=512):

    # 랜드마크를 활용하여 얼굴을 검출할 때 사용하는 변수
    face_img = src_img.copy()

    # 조합한 부위를 붙여넣기 위한 변수
    final_img = trg_img.copy()

    # 랜드마크 사용하기 위한 얼굴 검출 변수
    detector = dlib.get_frontal_face_detector()                                 # 이미지에서 얼굴을 찾는 검출기
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   # 68개의 랜드마크 예측기

    # 원본 이미지에서 얼굴을 검출한다.
    faces = detector(face_img)
    if len(faces) == 0:
        print("Error! 검출된 얼굴이 없습니다. 다른 사진을 이용해주세요.")
        return

    # 첫번째 검출된 얼굴만 활용하여 조합을 진행한다.
    face = faces[0]
    # 검출된 얼굴 이미지에서 68개의 랜드마크를 예측한다.
    landmarks = predictor(face_img, face)

    mask_img = np.zeros((face_img.shape[0], face_img.shape[1], 3), dtype=np.uint8) * 255

    # 세그멘테이션 조합일 경우
    if is_PIL == True:
        # print("face_img를 PIL 이미지로 변경!")
        face_img = util.cvtopil(face_img)
        # face_img.show()

        # 부위이미지가 투명도를 포함한 png 이미지(PIL Image)일 경우, 마스크 이미지도 PIL 이미지 형태로 바꿔줘야 한다.
        mask_img = util.cvtopil(mask_img)

    # set x, y padding
    if img_size == 512:
        # asian image padding (512x512 size)
        brow_x, brow_y = 10, 10
        eye_x, eye_y = 10, 5
        nose_x, nose_y = 20, 5
        mouth_x, mouth_y = 10, 10
    else:
        # white image padding (1024x1024 size)
        brow_x, brow_y = 10, 10
        eye_x, eye_y = 20, 20
        nose_x, nose_y = 40, 5
        mouth_x, mouth_y = 10, 10


    # 선택된 부위만 조합하기
    # 오른쪽 눈
    if trg_righteye is not 'None':

        w_righteye = util.make_white_img(trg_righteye, is_PIL)

        final_img = attach_features(final_img, trg_righteye, landmarks.part(36).x - 15, landmarks.part(39).x,
                                    round((landmarks.part(37).y + landmarks.part(38).y) / 2) - 10,
                                    round((landmarks.part(41).y + landmarks.part(40).y) / 2), eye_x,
                                    eye_y, is_PIL)

        # 오른쪽 눈(흰색) - 마스크 이미지에 붙여넣기
        mask_img = attach_features(mask_img, w_righteye, landmarks.part(36).x - 15, landmarks.part(39).x,
                                   round((landmarks.part(37).y + landmarks.part(38).y) / 2) - 10,
                                   round((landmarks.part(41).y + landmarks.part(40).y) / 2), eye_x,
                                   eye_y, is_PIL)

        # [추가] 왼쪽 눈
        if trg_lefteye is not 'None':

            w_lefteye = util.make_white_img(trg_lefteye, is_PIL)

            # 왼쪽 눈
            final_img = attach_features(final_img, trg_lefteye, landmarks.part(42).x, landmarks.part(45).x + 15,
                                        round((landmarks.part(43).y + landmarks.part(44).y) / 2) - 10,
                                        round((landmarks.part(47).y + landmarks.part(46).y) / 2), eye_x,
                                        eye_y, is_PIL)

            # 왼쪽 눈(흰색) - 마스크 이미지에 붙여넣기
            mask_img = attach_features(mask_img, w_lefteye, landmarks.part(42).x, landmarks.part(45).x + 15,
                                       round((landmarks.part(43).y + landmarks.part(44).y) / 2) - 10,
                                       round((landmarks.part(47).y + landmarks.part(46).y) / 2), eye_x,
                                       eye_y, is_PIL)

    # 오른쪽 눈썹
    if trg_righteyebrow is not 'None':

        # [추가] 후처리에 필요한 마스크 이미지 만들기
        # 부위 이미지를 흰색으로 만든다.
        w_righteyebrow = util.make_white_img(trg_righteyebrow, is_PIL)
        # w_lefteyebrow = util.make_white_img(trg_lefteyebrow, is_PIL)

        # 오른쪽 눈썹
        if landmarks.part(17).y > landmarks.part(21).y:     # 눈썹 끝이 눈썹 앞머리보다 내려온 경우
            final_img = attach_features(final_img, trg_righteyebrow, landmarks.part(17).x, landmarks.part(21).x,
                                      landmarks.part(19).y,
                                      landmarks.part(17).y, brow_x, brow_y, is_PIL)
            # 오른쪽 눈썹(흰색) - 마스크 이미지에 붙여넣기
            mask_img = attach_features(mask_img, w_righteyebrow, landmarks.part(17).x, landmarks.part(21).x,
                                       landmarks.part(19).y,
                                       landmarks.part(17).y, brow_x, brow_y, is_PIL)
        else:
            final_img = attach_features(final_img, trg_righteyebrow, landmarks.part(17).x, landmarks.part(21).x,
                                      landmarks.part(19).y,
                                      landmarks.part(21).y, brow_x, brow_y, is_PIL)
            mask_img = attach_features(mask_img, w_righteyebrow, landmarks.part(17).x, landmarks.part(21).x,
                                       landmarks.part(19).y,
                                       landmarks.part(21).y, brow_x, brow_y, is_PIL)

    # 왼쪽 눈썹
    if trg_righteyebrow is not 'None':

        # [추가] 후처리에 필요한 마스크 이미지 만들기
        # 부위 이미지를 흰색으로 만든다.
        w_lefteyebrow = util.make_white_img(trg_lefteyebrow, is_PIL)

        # 왼쪽 눈썹
        if landmarks.part(26).y > landmarks.part(22).y:     # 눈썹 끝이 눈썹 앞머리보다 내려온 경우
            final_img = attach_features(final_img, trg_lefteyebrow, landmarks.part(22).x, landmarks.part(26).x,
                                      landmarks.part(24).y,
                                      landmarks.part(26).y, brow_x, brow_y, is_PIL)
            # 왼쪽 눈썹(흰색) - 마스크 이미지에 붙여넣기
            mask_img = attach_features(mask_img, w_lefteyebrow, landmarks.part(22).x, landmarks.part(26).x,
                                       landmarks.part(24).y,
                                       landmarks.part(26).y, brow_x, brow_y, is_PIL)
        else:
            final_img = attach_features(final_img, trg_lefteyebrow, landmarks.part(22).x, landmarks.part(26).x,
                                      landmarks.part(24).y,
                                      landmarks.part(22).y, brow_x, brow_y, is_PIL)
            mask_img = attach_features(mask_img, w_lefteyebrow, landmarks.part(22).x, landmarks.part(26).x,
                                       landmarks.part(24).y,
                                       landmarks.part(22).y, brow_x, brow_y, is_PIL)

    # 코
    # x_padding : 40 -> 20
    if trg_nose is not 'None':

        w_nose = util.make_white_img(trg_nose, is_PIL)

        if is_seg == False:
            # 후처리 테스트 중, 눈과 겹치는 부분이 생기면 경계선이 티남 -> 눈 부위와 겹치지 않도록 수정
            # y 시작점 수정 : landmarks.part(27).y - 20  ==> landmarks.part(28).y
            final_img = attach_features(final_img, trg_nose, landmarks.part(31).x, landmarks.part(35).x,
                                      landmarks.part(28).y+20, landmarks.part(33).y, nose_x, nose_y, is_PIL)

            # 코(흰색) - 마스크 이미지에 붙여넣기
            # 후처리 테스트 중, 눈과 겹치는 부분이 생기면 경계선이 티남 -> 눈 부위와 겹치지 않도록 수정
            # y 시작점 수정 : landmarks.part(27).y - 20  ==> landmarks.part(28).y
            mask_img = attach_features(mask_img, w_nose, landmarks.part(31).x, landmarks.part(35).x,
                                       landmarks.part(28).y+20, landmarks.part(33).y, nose_x, nose_y, is_PIL)

        else:
            # 세그멘테이션일 경우에는 28번 랜드마크 사용하여 콧대 길게 잡기
            # 후처리 테스트 중, 눈과 겹치는 부분이 생기면 경계선이 티남 -> 눈 부위와 겹치지 않도록 수정
            # y 시작점 수정 : landmarks.part(27).y - 20  ==> landmarks.part(28).y
            final_img = attach_features(final_img, trg_nose, landmarks.part(31).x, landmarks.part(35).x,
                                       landmarks.part(27).y - 20, landmarks.part(33).y, nose_x, nose_y, is_PIL)

            # 코(흰색) - 마스크 이미지에 붙여넣기
            # 후처리 테스트 중, 눈과 겹치는 부분이 생기면 경계선이 티남 -> 눈 부위와 겹치지 않도록 수정
            # y 시작점 수정 : landmarks.part(27).y - 20  ==> landmarks.part(28).y
            mask_img = attach_features(mask_img, w_nose, landmarks.part(31).x, landmarks.part(35).x,
                                       landmarks.part(27).y - 20, landmarks.part(33).y, nose_x, nose_y, is_PIL)

    # 입
    # x_padding : 10 -> 20
    # y_padding : 10 -> 20
    if trg_mouth is not 'None':

        w_mouth = util.make_white_img(trg_mouth, is_PIL)

        final_img = attach_features(final_img, trg_mouth, landmarks.part(48).x, landmarks.part(54).x, landmarks.part(52).y,
                                  landmarks.part(57).y, mouth_x, mouth_y, is_PIL)

        # 입 마스크(흰색) - 이미지에 붙여넣기
        mask_img = attach_features(mask_img, w_mouth, landmarks.part(48).x, landmarks.part(54).x, landmarks.part(52).y,
                                   landmarks.part(57).y, mouth_x, mouth_y, is_PIL)

    return final_img, mask_img