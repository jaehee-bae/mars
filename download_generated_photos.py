# 설  명  : generated photos open API를 활용해 원하는 특징의 얼굴 이미지를 다운로드 하는 코드

import json
import urllib.request
import time

if __name__ == "__main__":

    # 호출 시 필요한 변수 선언
    page_no = 130               # 다운받을 이미지 페이지 시작번호
    page_no_last = 137          # 페이지 끝 번호
    version_no = 3              # 이미지 버전 (3이 가장 최신)
    api_key = ""                # 발급받은 API

    # 이미지 저장 경로
    SAVE_DIR = 'data/face/'

    # 페이지별로 반복하면서 데이터 다운로드
    for pno in range(page_no, page_no_last+1):

        print("===== " + str(pno) + " page start! =====")
        # 한 페이지 당 소요되는 시간 체크
        s_time = time.time()

        # open API 호출
        req = urllib.request.Request("https://api.generated.photos/api/v1/faces?api_key="+api_key+"&emotion=neutral&ethnicity=asian&per_page=100&version="
                                     +str(version_no)+"&page="+str(pno))
        data = urllib.request.urlopen(req).read()

        # bytes 타입의 데이터를 string 타입으로 변경
        data = data.decode('utf-8')
        print(data)

        # JSON 파싱
        jObject = json.loads(str(data))
        faces = jObject.get("faces")

        # 저장할 때 사용할 이미지 id 값
        img_no = 1

        for img in faces:
            img_id = img['id']
            img_url = img['urls'][4]['512']
            gender = img['meta']['gender'][0]
            hair_len = img['meta']['hair_length'][0]

            print(img_url)

            # 이미지 다운받기
            urllib.request.urlretrieve(img_url, SAVE_DIR + img_id + ".jpg")
            print("img_no : ", str(img_no), " / img_id : ", str(img_id))

            # 다운로드 중 연결이 끊어지는 현상이 자주 발생해서 넣어준 코드, 별 효과는 없는 것 같다.
            time.sleep(1)

            img_no = img_no + 1

        print("=== " + str(pno) + " 페이지 걸린 시간 : " + str(time.time()-s_time) + " ===")
