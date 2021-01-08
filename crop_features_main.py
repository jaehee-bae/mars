# 작성자  : 배재희
# 설  명  : 원본 이미지에서 부위 이미지 네모로 크롭해서 저장하는 코드

import dlib
import utils as util
import cv2
import glob
import os

# << 랜드마크를 활용하여 네모 형태로 부위이미지를 자르는 함수 >>
# 오른쪽 눈과 눈썹만 잘랐는데 왼쪽 눈과 눈썹도 추가
def crop_features_rec(src_img):

    face_img = src_img.copy()

    # 랜드마크 사용하기 위한 얼굴 검출 변수
    detector = dlib.get_frontal_face_detector()  # 이미지에서 얼굴을 찾는 검출기
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
    # assemble_face_features.py 의 assemble_features 함수와 동일한 x,y 좌표 값과 padding 값을 가져야 한다.

    # 오른쪽 눈썹
    if landmarks.part(17).y > landmarks.part(21).y:  # 눈썹 끝이 눈썹 앞머리보다 내려온 경우
        r_eyebrow = util.crop_rec_img(face_img, landmarks.part(17).x, landmarks.part(21).x, landmarks.part(19).y,landmarks.part(17).y, 10, 10)
    else :
        r_eyebrow = util.crop_rec_img(face_img, landmarks.part(17).x, landmarks.part(21).x, landmarks.part(19).y,landmarks.part(21).y, 10, 10)

    # [추가] 왼쪽 눈썹
    if landmarks.part(26).y > landmarks.part(22).y:
        l_eyebrow = util.crop_rec_img(face_img, landmarks.part(22).x, landmarks.part(26).x, landmarks.part(24).y,landmarks.part(26).y, 10, 10)
    else :
        l_eyebrow = util.crop_rec_img(face_img, landmarks.part(22).x, landmarks.part(26).x, landmarks.part(24).y,landmarks.part(22).y, 10, 10)


    # 오른쪽 눈
    # x_padding : 20 -> 10  y_padding : 20 -> 5
    r_eye = util.crop_rec_img(face_img, landmarks.part(36).x - 15, landmarks.part(39).x,
                                round((landmarks.part(37).y + landmarks.part(38).y) / 2) - 10,
                                round((landmarks.part(41).y + landmarks.part(40).y) / 2), 10,5)

    # [추가] 왼쪽 눈
    l_eye = util.crop_rec_img(face_img, landmarks.part(42).x, landmarks.part(45).x + 15,
                                    round((landmarks.part(43).y + landmarks.part(44).y) / 2) - 10,
                                    round((landmarks.part(47).y + landmarks.part(46).y) / 2), 10,5)

    # 코
    # x_padding : 40 -> 20
    nose = util.crop_rec_img(face_img, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(28).y + 20, landmarks.part(33).y, 20, 5)

    # 입
    mouth = util.crop_rec_img(face_img, landmarks.part(48).x, landmarks.part(54).x, landmarks.part(52).y,
                                landmarks.part(57).y, 10, 10)

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

    type = 'val'
    save_dir = 'C:/Users/jaehee/Documents/210106/pix_asian1_features/add1/' + type + '/'

    # images = glob.glob('sample_img/original/502/*.jpg')
    images = glob.glob('C:/Users/jaehee/Documents/210106/pix_asian1/add1/'+ type +'/*.jpg')
    print(images)

    for fname in images:

        # 저장 시 사용하기 위한 원본 얼굴 이미지 파일명 (여러 장일 경우)
        # img_name = '019883'
        img_name = fname.split('\\')[1].split('.')[0]
        # print(img_name)
        # img_name = '13'

        # 원본 얼굴 이미지
        img = cv2.imread(fname)
        # cv2.imshow("original", img)

        # 원본 사진에서 부위 별 이미지 자르는 함수
        r_eye, l_eye, r_eyebrow, l_eyebrow, nose, mouth = crop_features_rec(img)
        # cv2.imshow("r_eye", r_eye)
        # cv2.imshow("l_eye", l_eye)
        # cv2.imshow("r_eyebrow", r_eyebrow)
        # cv2.imshow("l_eyebrow", l_eyebrow)
        # cv2.imshow("nose", nose)
        # cv2.imshow("mouth", mouth)


        # 얼굴부위 폴더 생성
        createFolder(save_dir + 'reye')
        createFolder(save_dir + 'reyebrow')
        createFolder(save_dir + 'nose')
        createFolder(save_dir + 'mouth')


        # 부위이미지 저장 -> 데이터 확인하고 3개만 선정할 예정
        # 이상한 부위 이미지 골라내기 위해 reye, reyebrow 폴더에 같이 저장
        cv2.imwrite(save_dir + 'reye/' + img_name + '_reye.jpg', r_eye)
        cv2.imwrite(save_dir + 'reye/' + img_name + '_leye.jpg', l_eye)

        cv2.imwrite(save_dir + 'reyebrow/' + img_name + '_reyebrow.jpg', r_eyebrow)
        cv2.imwrite(save_dir + 'reyebrow/' + img_name + '_leyebrow.jpg', l_eyebrow)

        cv2.imwrite(save_dir + 'nose/' + img_name + '_nose.jpg', nose)
        cv2.imwrite(save_dir + 'mouth/' + img_name + '_mouth.jpg', mouth)
        print(img_name, "저장완료!")


    cv2.waitKey(0)
    cv2.destroyAllWindows()
