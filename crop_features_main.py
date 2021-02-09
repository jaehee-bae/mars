# 설  명  : 원본 이미지에서 부위 이미지 네모로 크롭해서 저장하는 코드

import dlib
import cv2
import glob
import os

# << 원본 이미지와 좌표값을 전달 받아서 네모로 자르는 함수 >>
def crop_rec_img(img, start_x, last_x, start_y, last_y, x_padding, y_padding):
    crop_img = img[start_y-y_padding : last_y+y_padding, start_x-x_padding:last_x+x_padding]
    return crop_img

# << 랜드마크를 활용하여 네모 형태로 얼굴 부위이미지(왼/오 눈썹, 왼/오 눈, 코, 입)를 자르는 함수 >>
def crop_features_rec(src_img, img_size):

    face_img = src_img.copy()

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

    # 랜드마크 사용하기 위한 얼굴 검출 변수
    detector = dlib.get_frontal_face_detector()                                # 이미지에서 얼굴을 찾는 검출기
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68개의 랜드마크 예측기

    # 원본 이미지에서 얼굴을 검출한다.
    faces = detector(face_img)
    if len(faces) == 0:
        print("Error! 검출된 얼굴이 없습니다. 다른 사진을 이용해주세요.")
        return

    # 첫번째 검출된 얼굴만 활용하여 조합을 진행한다.
    face = faces[0]
    # 검출된 얼굴 이미지에서 68개의 랜드마크를 예측한다.
    landmarks = predictor(face_img, face)

    # 부위 별로 자르기
    # x, y의 시작과 마지막 좌표 값을 가지고 원본 이미지에서 자른다.
    # 얼굴부위 조합 시에도 자를 때 사용한 x, y padding 값과 동일한 값을 사용해야 한다.

    # 오른쪽 눈썹
    if landmarks.part(17).y > landmarks.part(21).y:  # 눈썹 끝이 눈썹 앞머리보다 내려온 경우
        r_eyebrow = crop_rec_img(face_img, landmarks.part(17).x, landmarks.part(21).x, landmarks.part(19).y,
                                 landmarks.part(17).y, brow_x, brow_y)
    else:
        r_eyebrow = crop_rec_img(face_img, landmarks.part(17).x, landmarks.part(21).x, landmarks.part(19).y,
                                 landmarks.part(21).y, brow_x, brow_y)

    # [추가] 왼쪽 눈썹
    if landmarks.part(26).y > landmarks.part(22).y:
        l_eyebrow = crop_rec_img(face_img, landmarks.part(22).x, landmarks.part(26).x, landmarks.part(24).y,
                                 landmarks.part(26).y, brow_x, brow_y)
    else:
        l_eyebrow = crop_rec_img(face_img, landmarks.part(22).x, landmarks.part(26).x, landmarks.part(24).y,
                                 landmarks.part(22).y, brow_x, brow_y)

    # 오른쪽 눈
    r_eye = crop_rec_img(face_img, landmarks.part(36).x - 15, landmarks.part(39).x,
                         round((landmarks.part(37).y + landmarks.part(38).y) / 2) - 10,
                         round((landmarks.part(41).y + landmarks.part(40).y) / 2), eye_x, eye_y)

    # [추가] 왼쪽 눈
    l_eye = crop_rec_img(face_img, landmarks.part(42).x, landmarks.part(45).x + 15,
                         round((landmarks.part(43).y + landmarks.part(44).y) / 2) - 10,
                         round((landmarks.part(47).y + landmarks.part(46).y) / 2), eye_x,
                         eye_y)

    # 코
    nose = crop_rec_img(face_img, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(28).y + 20,
                        landmarks.part(33).y, nose_x, nose_y)

    # 입
    mouth = crop_rec_img(face_img, landmarks.part(48).x, landmarks.part(54).x, landmarks.part(52).y,
                         landmarks.part(57).y, mouth_x, mouth_y)

    return r_eye, l_eye, r_eyebrow, l_eyebrow, nose, mouth


# << 디렉토리에 필요한 폴더가 없으면 생성하는 함수 >>
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            # print(directory + ' 생성 완료!')
    except OSError:
        print('Error: Creating directory. ' + directory)


# << main >>
if __name__ == "__main__":

    # 원본 얼굴 이미지가 존재하는 경로
    IMG_DIR = 'data/face/*.jpg'
    # 얼굴부위 이미지를 저장할 경로
    SAVE_DIR = 'data/features/'

    images = glob.glob(IMG_DIR)

    # 이미지 파일명을 얻기위해 필요
    # 우분투일 경우, '/'
    # 윈도우일 경우, '\\'
    slush_count = IMG_DIR.count('/') + 1

    for fname in images:

        # 저장 시 사용하기 위한 원본 얼굴 이미지 파일명 (여러 장일 경우)
        img_name = fname.split('/')[slush_count].split('.')[0]

        # 원본 얼굴 이미지
        img = cv2.imread(fname)
        # cv2.imshow("original", img)

        # 원본 사진에서 부위 별 이미지 자르는 함수
        r_eye, l_eye, r_eyebrow, l_eyebrow, nose, mouth = crop_features_rec(img, 512)
        # cv2.imshow("r_eye", r_eye)
        # cv2.imshow("l_eye", l_eye)
        # cv2.imshow("r_eyebrow", r_eyebrow)
        # cv2.imshow("l_eyebrow", l_eyebrow)
        # cv2.imshow("nose", nose)
        # cv2.imshow("mouth", mouth)


        # 얼굴부위 폴더 생성
        createFolder(SAVE_DIR + 'reye')
        createFolder(SAVE_DIR + 'reyebrow')
        createFolder(SAVE_DIR + 'nose')
        createFolder(SAVE_DIR + 'mouth')

        # 부위이미지 저장
        # 이상한 부위 이미지 골라내기 위해 leye, leyebrow를 reye, reyebrow 폴더에 같이 저장
        cv2.imwrite(SAVE_DIR + 'reye/' + img_name + '_reye.jpg', r_eye)
        cv2.imwrite(SAVE_DIR + 'reye/' + img_name + '_leye.jpg', l_eye)

        cv2.imwrite(SAVE_DIR + 'reyebrow/' + img_name + '_reyebrow.jpg', r_eyebrow)
        cv2.imwrite(SAVE_DIR + 'reyebrow/' + img_name + '_leyebrow.jpg', l_eyebrow)

        cv2.imwrite(SAVE_DIR + 'nose/' + img_name + '_nose.jpg', nose)
        cv2.imwrite(SAVE_DIR + 'mouth/' + img_name + '_mouth.jpg', mouth)
        print(img_name, "저장완료!")


    cv2.waitKey(0)
    cv2.destroyAllWindows()
