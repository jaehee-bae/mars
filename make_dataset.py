# 얼굴부위 조합 ~ 후 처리(포아송) ~ 채색을 위한 paired dataset을 생성하는 코드

import cv2
import glob
import numpy as np
import json

import assemble_face_features as makeface
import faceparsing as fparsing
import poissonimageediting as poisson
import binary_filterlist as bfilter
import utils as util

FACE_DIR = 'data/face/'             # 원본 이미지 저장경로
FEATURE_DIR = 'data/features/'      # 크롭한 얼굴부위 이미지 저장경로
SAVE_DIR = 'dataset/'               # 결과 이미지 저장경로

images = glob.glob(FACE_DIR + '*.jpg')

# 이미지 파일명을 얻기위해 필요
# 우분투일 경우, '/'
# 윈도우일 경우, '\\'
slush_count = FACE_DIR.count('/') + 1

for fname in images:

    # === 필요한 변수 선언 ===
    # 원본 얼굴 이미지 파일명
    img_name = fname.split('/')[slush_count].split('.')[0]

    # 원본 얼굴 이미지
    org_img = cv2.imread(fname)

    # image resize
    org_img = cv2.resize(org_img, (512,512))
    img = org_img.copy()

    # 부위 이미지 불러오기
    # load_face_features : 오른쪽 눈, 눈썹, 코, 입 이미지 불러오는 함수
    # load_face_features_detail : 양쪽 눈, 눈썹, 코, 입 이미지 불러오는 함수
    # eye, eyebrow, nose, mouth = makeface.load_face_features('eye'), makeface.load_face_features('eyebrow'), makeface.load_face_features('nose'), makeface.load_face_features('mouth')
    eye, leye, reye_nm, leye_nm = makeface.load_face_features_detail(FEATURE_DIR, 'eye')
    eyebrow, leyebrow, reyebrow_nm, leyebrow_nm = makeface.load_face_features_detail(FEATURE_DIR, 'eyebrow')
    nose, _, nose_nm, _ = makeface.load_face_features_detail(FEATURE_DIR, 'nose')
    mouth, _, mouth_nm, _ = makeface.load_face_features_detail(FEATURE_DIR, 'mouth')

    # test features images
    # cv2.imshow("eye", eye)
    # cv2.imshow("leye", leye)
    # print(reye_nm, leye_nm)
    # cv2.imshow("eyebrow", eyebrow)
    # cv2.imshow("leyebrow", leyebrow)
    # print(reyebrow_nm, leyebrow_nm)
    # cv2.imshow("nose", nose)
    # cv2.imshow("mouth", mouth)
    # print(nose_nm, mouth_nm)

    # [추가] 헤어 영역 PIL 이미지 가져오기 (세그멘테이션 활용)
    hair_img = fparsing.get_hair_area(img, 512)

    # [추가] 검정색 픽셀을 찾아서 투명으로 변경
    pil_hair_img = util.make_transparent_img(hair_img, r=0, g=0, b=0)

    # 1차 조합 - 원본에 입만 잘라붙인 이미지
    first_assemble, first_mask = makeface.assemble_features_detail(trg_mouth=mouth, src_img=img, is_PIL=False, trg_img=img)

    # 2차 조합 - 원본에 눈썹, 코 잘라붙인 이미지
    second_assemble, second_mask = makeface.assemble_features_detail(trg_righteyebrow=eyebrow, trg_nose=nose, src_img=img,
                                                         is_PIL=False, trg_img=img, trg_lefteyebrow=leyebrow)

    # 3차 조합 - 원본에 눈만 잘라붙인 이미지
    third_assemble, third_mask = makeface.assemble_features_detail(trg_righteye=eye, src_img=img, is_PIL=False,
                                                                        trg_img=img, trg_lefteye=leye)
    # === 필요한 변수 선언 끝 ===

    # 1차 후처리 - mouth
    src = np.array(first_assemble / 255.0, dtype=np.float32)
    tar = np.array(img / 255.0, dtype=np.float32)
    first_mask = np.array(cv2.cvtColor(first_mask, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    ret, first_mask = cv2.threshold(first_mask, 0, 255, cv2.THRESH_OTSU)

    first_blended, overlapped = poisson.poisson_blend(src, first_mask / 255.0, tar, 'import', SAVE_DIR)
    print("[", img_name, "  - 1차 (mouth) 후처리 완료]")


    # 2차 후처리 - 눈썹, 코
    src = np.array(second_assemble / 255.0, dtype=np.float32)
    tar = np.array(first_blended / 255.0, dtype=np.float32)
    second_mask = np.array(cv2.cvtColor(second_mask, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    ret, second_mask = cv2.threshold(second_mask, 0, 255, cv2.THRESH_OTSU)

    second_blended, overlapped = poisson.poisson_blend(src, second_mask / 255.0, tar, 'import', SAVE_DIR)
    print("[", img_name, "  - 2차 (눈썹, 코) 후처리 완료]")


    # 3차 후처리 - 눈
    src = np.array(third_assemble / 255.0, dtype=np.float32)
    tar = np.array(second_blended / 255.0, dtype=np.float32)
    third_mask = np.array(cv2.cvtColor(third_mask, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    ret, third_mask = cv2.threshold(third_mask, 0, 255, cv2.THRESH_OTSU)

    final_blended, overlapped = poisson.poisson_blend(src, third_mask / 255.0, tar, 'import', SAVE_DIR)
    print("[", img_name, "  - 3차 (눈) 후처리 완료]")


    # 최종 이미지에 머리카락 덮기 - 튀어나간 눈썹 영역 없애기 위함
    final_blended = util.cvtopil(final_blended)
    final_blended.paste(pil_hair_img, (0,0), pil_hair_img)

    # opencv 이미지로 변경 후, 이진화 필터 적용
    final_blended = util.piltocv(final_blended)

    # 서양인 얼굴 학습 시 사용한 필터 : 30번 이진화 필터 (은진씨가 만듦)
    # f_face = bfilter.binary_filter(30, final_blended)

    # 동양인 얼굴 생성 시 사용한 필터 : 32번 이진화 필터
    f_face = bfilter.binary_filter(32, final_blended)


    # ==== 결과 이미지 저장하기 ====

    # 단계 별 산출물 모두 저장하기 (발표용, 포트폴리오용)
    # 머리카락 영역 이미지 저장하기 (png)
    pil_hair_img.save(SAVE_DIR + img_name + '_hair.png')
    # 1차 후처리(입)를 위한 조합, 마스크 이미지
    cv2.imwrite(SAVE_DIR + img_name + '_first_assemble.jpg', first_assemble)
    cv2.imwrite(SAVE_DIR + img_name + '_first_mask.jpg', first_mask)
    # 2차 후처리(눈썹, 코)를 위한 조합, 마스크 이미지
    cv2.imwrite(SAVE_DIR + img_name + '_second_assemble.jpg', second_assemble)
    cv2.imwrite(SAVE_DIR + img_name + '_second_mask.jpg', second_mask)
    # 3차 후처리(눈)를 위한 조합, 마스크 이미지
    cv2.imwrite(SAVE_DIR + img_name + '_third_assemble.jpg', third_assemble)
    cv2.imwrite(SAVE_DIR + img_name + '_third_mask.jpg', third_mask)


    # 원본 사진과 비교하기 위한 이미지 (원본 + 후처리 + 이진화)
    con_img = cv2.hconcat([org_img, final_blended, f_face])
    test_img = cv2.hconcat([f_face, final_blended])

    # 테스트
    # cv2.imshow("final_blended", final_blended)  # 최종 후처리 이미지
    # cv2.imshow("f_face", f_face)                # 최종 이진화 이미지
    # cv2.imshow("con_img", con_img)              # 비교 이미지

    # 저장, 저장하기 위한 폴더가 미리 만들어져 있어야 한다.
    cv2.imwrite(SAVE_DIR + img_name + '_poisson.jpg', final_blended)
    cv2.imwrite(SAVE_DIR + img_name + '_binary.jpg', f_face)
    cv2.imwrite(SAVE_DIR + 'merged/' + img_name + '_merged.jpg', con_img)
    cv2.imwrite(SAVE_DIR + 'test/' + img_name + '_test.jpg', test_img)

    # 얼굴조합 시, 선택한 부위 이미지 번호도 json 파일로 같이 저장
    # reye_nm, leye_nm, reyebrow_nm, leyebrow_nm, nose_nm, mouth_nm
    json_filepath = SAVE_DIR + img_name + '.json'

    data = {}
    data['imginfo'] = []
    data['imginfo'].append({
        "img": img_name,
        "reye_nm": reye_nm,
        "leye_nm": leye_nm,
        "reyebrow_nm": reyebrow_nm,
        "leyebrow_nm": leyebrow_nm,
        "nose_nm": nose_nm,
        "mouth_nm": mouth_nm
    })

    # json 파일로 저장 (indent 옵션은 출력 시, 보기 좋게 하기 위함)
    with open(json_filepath, "w") as outfile:
        json.dump(data, outfile, indent=4)

    print("<< ", img_name, " - 저장완료! >>")

    # ==== 결과 이미지 저장하기 끝 ====

cv2.waitKey(0)
cv2.destroyAllWindows()
