# 작성자  : 배재희
# 설  명  : 부위 형태에 맞게 크롭하는 코드 (랜드마크, 세그멘테이션)
# 주의 사항
# 1) dlib의 랜드마크를 사용하기 때문에 landmark(shape) predictor가 필요 (shape_predictor_68_face_landmarks.dat 파일)
# 2) 원본 이미지의 경우, 랜드마크를 통해 얼굴을 검출했을 때 하나의 얼굴만 검출되야 하며 부위 별 랜드마크가 올바른 위치에 잘 검출되야 어색하지 않은 새로운 얼굴을 만들 수 있음
#    (그렇지 않으면 이상한 위치에 부위 이미지가 배치되어 어색한 결과가 나옴)
# 참고 자료
# 랜드마크 - https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/#pyi-pyimagesearch-plus-pricing-modal
# 세그멘테이션 - https://github.com/zllrunning/face-parsing.PyTorch

import cv2
import glob
import numpy as np
from PIL import Image
import utils as util

# 랜드마크 모델
from collections import OrderedDict
import dlib

# 세그멘테이션 모델
from model import BiSeNet
import torch
import torchvision.transforms as transforms

# define a dictionary that maps the indexes of the facial landmarks to specific face regions
# (부위명, (랜드마크 시작점, 랜드마크 마지막점))
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17)),
    ("outer_lips", (48, 59))       # 추가
])

# << 랜드마크를 활용하여 형태에 맞게 부위이미지를 자르는 함수 >>
def crop_features_landmark(src_img):
    # 랜드마크 사용하기 위한 얼굴 검출 변수
    detector = dlib.get_frontal_face_detector()                                # 이미지에서 얼굴을 찾는 검출기
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68개의 랜드마크 예측기

    # 원본 이미지에서 얼굴을 검출한다.
    faces = detector(src_img)
    if len(faces) == 0:
        print("Error! 검출된 얼굴이 없습니다. 다른 사진을 이용해주세요.")
        return

    # 첫번째 검출된 얼굴만 활용하여 조합을 진행한다.
    face = faces[0]
    # 검출된 얼굴 이미지에서 68개의 랜드마크를 예측한다.
    landmarks = predictor(src_img, face)

    # 68개의 x, y 좌표를 저장하기 위한 변수
    shape = np.empty([68, 2], dtype=int)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        # 왼쪽, 오른쪽 눈일 경우 y 값 padding 준 값으로 수정하여 크롭 (안그러면 윗쪽 부분이 잘림)
        if n == 37 or n == 38 or n == 43 or n == 44:
            shape[n][0] = x
            shape[n][1] = y - 10
        else:
            shape[n][0] = x
            shape[n][1] = y

    # 얼굴 부위 특징 별로 저장
    for(i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # j : 랜드마크 시작점, k : 랜드마크 마지막점
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        # pts : 부위 별 랜드마크의 (x, y) 좌표 값을 저장하고 있는 NumPy 배열
        pts = shape[j:k]

        if name == "right_eye":             # 점을 연결해서 다각형을 만들고 크롭한다.
            hull = cv2.convexHull(pts)      # convexHull : 주어진 점을 둘러싸는 다각형을 구하는 알고리즘 함수
            overlay_img = src_img.copy()
            righteye = util.mask_and_crop(overlay_img, hull)

        elif name == "right_eyebrow":
            hull = cv2.convexHull(pts)
            overlay_img = src_img.copy()
            righteyebrow = util.mask_and_crop(overlay_img, hull)

        elif name == "nose":
            hull = cv2.convexHull(pts)
            overlay_img = src_img.copy()
            nose = util.mask_and_crop(overlay_img, hull)

        elif name == "mouth":
            hull = cv2.convexHull(pts)
            overlay_img = src_img.copy()
            mouth = util.mask_and_crop(overlay_img, hull)

    return righteye, righteyebrow, nose, mouth

# << 랜드마크를 활용하여 입 영역을 다른 색으로 칠하여 리턴해주는 함수 >>
def mouth_landmark(src_img):
    # 랜드마크 사용하기 위한 얼굴 검출 변수
    detector = dlib.get_frontal_face_detector()                                # 이미지에서 얼굴을 찾는 검출기
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68개의 랜드마크 예측기

    # 리턴해 줄 이미지
    img = src_img.copy()

    # 원본 이미지에서 얼굴을 검출한다.
    faces = detector(src_img)
    if len(faces) == 0:
        print("Error! 검출된 얼굴이 없습니다. 다른 사진을 이용해주세요.")
        return

    # 첫번째 검출된 얼굴만 활용하여 조합을 진행한다.
    face = faces[0]
    # 검출된 얼굴 이미지에서 68개의 랜드마크를 예측한다.
    landmarks = predictor(src_img, face)

    # 68개의 x, y 좌표를 저장하기 위한 변수
    shape = np.empty([68, 2], dtype=int)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        # === 기존 랜드마크 기능을 그대로 사용하면 좌우 입꼬리 시작점 좌표를 잘 못잡아서 좌표 값 수정 ===
        # 입 꼬리일 경우 x 좌표 수정
        tail_padding_x = 20
        tail_padding_y = 10

        # 왼쪽 입꼬리
        # if n in (49, 50, 51):
        if n == 48:
            # print("n is 48")
            shape[n][0] = x - tail_padding_x
            shape[n][1] = y - tail_padding_y

        # 오른쪽 입꼬리
        # elif n in (55, 54, 53):
        elif n == 54:
            # print("n is 54")
            shape[n][0] = x + tail_padding_x
            shape[n][1] = y - tail_padding_y

        # outer edge of lips
        elif n in (49, 50, 51, 52, 53):
            shape[n][0] = x
            shape[n][1] = y - tail_padding_y

        elif n in (55, 56, 57, 58, 59):
            shape[n][0] = x
            shape[n][1] = y + tail_padding_y

        else:
            shape[n][0] = x
            shape[n][1] = y

        # ===

        # shape[n][0] = x
        # shape[n][1] = y


    # 얼굴 부위 특징 별로 저장
    for(i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # j : 랜드마크 시작점, k : 랜드마크 마지막점
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        # pts : 부위 별 랜드마크의 (x, y) 좌표 값을 저장하고 있는 NumPy 배열
        pts = shape[j:k]

        if name == "outer_lips":
            # hull = cv2.convexHull(pts)
            # overlay_img = src_img.copy()
            # mouth = util.mask_and_crop(overlay_img, hull)

            # 윤곽선 내부에 흰색 채우기
            # cv2.drawContours(img, [pts], -1, (255, 255, 255), -1)

            # 마우스의 컨투어 영역 저장
            mouth_pts = pts

    return mouth_pts
    # return righteye, righteyebrow, nose, mouth



# << 부위 별 인덱스 정보를 활용하여 부위 형태 이미지를 매핑하는(?) 함수 >>
def vis_parsing_maps(im, parsing_anno, stride):

    # PIL 이미지를 openCV 이미지로 변환
    im = util.piltocv(im)

    # 원본 얼굴 이미지 shape: (512, 512, 3)
    vis_im = im.copy().astype(np.uint8)

    # 부위 별 세그멘테이션 인덱스를 가지고 있는 2차원 배열
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    # 얼굴 부위 인덱스 최대 값
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)

        # 부위영역 픽셀의 x, y 좌표 값
        y = index[0]
        x = index[1]
        if (len(x) <= 0) or (len(y) <= 0):      # 인덱스에 해당하는 부위영역이 없을 때
            # if pi in (3, 5, 10, 12, 13):        # 오른쪽 눈, 오른쪽 눈썹, 코, 윗입술, 아랫입술에 해당하는 영역이 없을 때
            #     print("face feature (pi : ", pi, ") 을 찾을 수 없습니다. 다른 이미지를 이용하세요.")
            #     return

            # exp42는 입 부분만 필요함
            if pi in (12, 13):        # 오른쪽 눈, 오른쪽 눈썹, 코, 윗입술, 아랫입술에 해당하는 영역이 없을 때
                print("face feature (pi : ", pi, ") 을 찾을 수 없습니다. 다른 이미지를 이용하세요.")
                return

            # 눈과 눈썹의 경우, 양쪽을 같은 영역으로 찾았을 때 예외처리 필요함 (크기로 해결?)

            print('[pi->',pi,'] x or y is empty!')

        else:
            if pi == 12:                    # upper lip
                mouth_top_index_y = y
                mouth_top_index_x = x
            elif pi == 13:                  # lower lip
                mouth_bottom_index_y = y
                mouth_bottom_index_x = x
            # elif pi == 10:                  # nose
            #     crop_nose = util.mask_and_crop_xy(vis_im, x, y, 2, 2)
            # elif pi == 3:                   # right_eyebrow
            #     crop_rightbrow = util.mask_and_crop_xy(vis_im, x, y, 2, 2)
            # elif pi == 5:                   # right_eye
            #     crop_righteye = util.mask_and_crop_xy(vis_im, x, y, 2, 2)

    crop_mouth = util.mask_and_crop_xy(vis_im, np.concatenate([mouth_top_index_x,mouth_bottom_index_x]), np.concatenate([mouth_top_index_y,mouth_bottom_index_y]), 0, 0)

    return crop_mouth
    # return crop_righteye, crop_rightbrow, crop_nose, crop_mouth

# << 학습된 세그멘테이션 모델을 이용하여 부위 영역을 찾고 형태에 맞춰 자르는 함수 >>
def crop_features_segmentation(src_img):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # 세그멘테이션 모델 파일 호출
    net.load_state_dict(torch.load('79999_iter.pth'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():

        # src_img를 PIL Image로 변경
        src_img = util.cvtopil(src_img)

        # 원본 이미지를 (512, 512)로 리사이즈
        # (Q) 리사이즈하면 이미지 깨지는 데 안하면 에러 나나?
        resize_img = src_img.resize((1024, 1024), Image.BILINEAR)
        src_img = to_tensor(resize_img)
        src_img = torch.unsqueeze(src_img, 0)
        src_img = src_img.cuda()
        out = net(src_img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # face feature (pi :  3 ) 을 찾을 수 없습니다. 다른 이미지를 이용하세요.
        # TypeError: 'NoneType' object is not iterable
        # righteye, righteyebrow, nose, mouth = vis_parsing_maps(resize_img, parsing, stride=1)
        mouth = vis_parsing_maps(resize_img, parsing, stride=1)

    # return righteye, righteyebrow, nose, mouth
    return mouth


# ==== 이진화 이미지 테스트 ====

# << 마스크할 영역의 x, y좌표 값 받아서 이진화 마스크 이미지를 만들어주는 함수 >>
def make_mask_img(img, x, y):
    # findcontours 함수에서 사용할 마스크 이미지
    # 넘파이 배열은 데이터타입이 디폴트로 np.int64 타입임 (dtype=np.uint8 명시해주지 않으면 디폴트로 생성됨)
    # 하지만 cv2.cvtColor 메소드는 8 bit나 1 bit의 데이터타입 필요함
    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

    # 남길 영역만 흰색으로 변경
    mask[y, x, :] = [255, 255, 255]

    # 그레이스케일 이미지로 변경 (findContours 때문에 필요)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 이진화 이미지로 변경
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask

# << 부위 별 인덱스 정보를 활용하여 얼굴을 찾고 윤곽선만 남기는 함수 >>
def parsing_face(im, parsing_anno, stride):

    # PIL 이미지를 openCV 이미지로 변환
    im = util.piltocv(im)

    # 원본 얼굴 이미지 shape: (512, 512, 3)
    vis_im = im.copy().astype(np.uint8)

    # 부위 별 세그멘테이션 인덱스를 가지고 있는 2차원 배열
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    # 흰 배경에 검은 윤곽선을 저장하고 있는 이미지
    # vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    line_img = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    # 파란색 배경으로 변경
    line_img[0:line_img.shape[0], 0:line_img.shape[1]] = [255, 0, 0]

    # 얼굴 부위 인덱스 최대 값
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)

        # 부위영역 픽셀의 x, y 좌표 값
        y = index[0]
        x = index[1]

        if (len(x) <= 0) or (len(y) <= 0):      # 인덱스에 해당하는 부위영역이 없을 때
            if pi == 1:                       # 피부 영역에 해당하는 인덱스가 없을 때
                print("face feature (pi : ", pi, ") 을 찾을 수 없습니다. 다른 이미지를 이용하세요.")
                return

        else:
            ## skin, neck, cloth, mouth, upper_lip, lower_lip, left_eye, right_eye, left_eyebrow, right_eyebrow => 입술 윤곽선이 필터로 추출이 어려워서 세그멘테이션로 윤곽선 잡기 테스트
            if pi in (1,14,16,11,12,13):
                # 선택된 영역의 x, y 좌표 값을 이용하여 마스크 이미지 생성
                mask = make_mask_img(line_img, x, y)

                # [추가] 원본에서 헤어 영역 찾기
                # 마스크 이미지에서 남길 영역의 픽셀만 원본 이미지 픽셀로 붙여넣기
                mask[y, x, :] = vis_im[y, x, :]

                # 컨투어 영역 찾고 라인 그리기
                line_img = util.find_and_draw_contourline(mask, line_img)
    # ========
    # 얼굴 라인에 맞춰서 컨투어 값이 제대로 잡혔는 지 테스트
    # vis_im = util.draw_contour_line(vis_im, hull)

    # 원본 이미지에 세그멘테이션으로 찾은 영역 오버레이
    # vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # 테스트
    # vis_im = cv2.resize(vis_im, (800,800))        ## detect 한 영역이 깨짐(neck)
    # cv2.imshow("vis_im", vis_im)
    # cv2.imshow("line_img", line_img)
    # cv2.imshow("neck_mask", neck_mask)

    return line_img
    # return crop_righteye, crop_rightbrow, crop_nose, crop_mouth

# << [추가] 세그멘테이션을 이용하여 원본(컬러)의 헤어영역 구하기 >>
def find_hairarea(im, parsing_anno, stride):
    # PIL 이미지를 openCV 이미지로 변환
    im = util.piltocv(im)

    # 원본 얼굴 이미지 shape
    vis_im = im.copy().astype(np.uint8)

    # 부위 별 세그멘테이션 인덱스를 가지고 있는 2차원 배열
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    # 흰 배경에 검은 윤곽선을 저장하고 있는 이미지
    mask_img = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3), dtype=np.uint8)

    # 얼굴 부위 인덱스 최대 값
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)

        # 부위영역 픽셀의 x, y 좌표 값
        y = index[0]
        x = index[1]

        if (len(x) <= 0) or (len(y) <= 0):  # 인덱스에 해당하는 부위영역이 없을 때
            if pi == 17:  # 헤어 영역에 해당하는 인덱스가 없을 때
                print("face feature (pi : ", pi, ") 을 찾을 수 없습니다. 다른 이미지를 이용하세요.")
                return

        else:
            if pi == 17:    # 헤어

                # cv2.imshow("vis_im", vis_im)
                # cv2.imshow("mask_img", mask_img)

                # 마스크 이미지에서 남길 영역의 픽셀만 원본 이미지 픽셀로 붙여넣기
                mask_img[y, x, :] = vis_im[y, x, :]

    return mask_img


# << 세그멘테이션을 활용하여 얼굴 선을 검출하는 함수 >>
def detect_face_line(src_img):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # 세그멘테이션 모델 파일 호출
    net.load_state_dict(torch.load('79999_iter.pth'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        # src_img를 PIL Image로 변경
        src_img = util.cvtopil(src_img)

        # 원본 이미지를 (512, 512)로 리사이즈
        # (Q) 리사이즈하면 이미지 깨지는 데 안하면 에러 나나?
        # resize_img = src_img.resize((512, 512), Image.BILINEAR)
        resize_img = src_img.resize((1024, 1024), Image.BILINEAR)
        src_img = to_tensor(resize_img)
        src_img = torch.unsqueeze(src_img, 0)
        src_img = src_img.cuda()
        out = net(src_img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # righteye, righteyebrow, nose, mouth = vis_parsing_maps(resize_img, parsing, stride=1)

        # 얼굴 선 검출
        line_img = parsing_face(resize_img, parsing, stride=1)

    return line_img

# << 세그멘테이션을 활용하여 헤어영역의 이미지를 가져오는 함수 >>
def get_hair_area(src_img):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # 세그멘테이션 모델 파일 호출
    net.load_state_dict(torch.load('79999_iter.pth'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        # src_img를 PIL Image로 변경
        src_img = util.cvtopil(src_img)

        resize_img = src_img.resize((1024, 1024), Image.BILINEAR)
        src_img = to_tensor(resize_img)
        src_img = torch.unsqueeze(src_img, 0)
        src_img = src_img.cuda()
        out = net(src_img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # 헤어영역 검출
        hair_img = find_hairarea(resize_img, parsing, stride=1)

    return hair_img


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
    r_eye = util.crop_rec_img(face_img, landmarks.part(36).x - 15, landmarks.part(39).x,
                                round((landmarks.part(37).y + landmarks.part(38).y) / 2) - 10,
                                round((landmarks.part(41).y + landmarks.part(40).y) / 2), 20,20)

    # [추가] 왼쪽 눈
    l_eye = util.crop_rec_img(face_img, landmarks.part(42).x, landmarks.part(45).x + 15,
                                    round((landmarks.part(43).y + landmarks.part(44).y) / 2) - 10,
                                    round((landmarks.part(47).y + landmarks.part(46).y) / 2), 20,
                                    20)

    # 코
    nose = util.crop_rec_img(face_img, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(28).y + 20, landmarks.part(33).y, 40, 5)

    # 입
    mouth = util.crop_rec_img(face_img, landmarks.part(48).x, landmarks.part(54).x, landmarks.part(52).y,
                                landmarks.part(57).y, 10, 10)

    return r_eye, l_eye, r_eyebrow, l_eyebrow, nose, mouth
