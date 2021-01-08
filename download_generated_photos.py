# 작성일 : 21.01.04
# generated photos openAPI를 활용해 얼굴 이미지를 다운로드 하는 코드
import json
import urllib.request
import time

if __name__ == "__main__":

    # 한 계정 당 50번 호출 무료
    # Download - 21,225
    # open API
    # v3 version : 56,366 // 564 pages
    # v2 version : 1,938
    # v1 version : 720

    # 호출 시 필요한 변수 선언
    page_no = 130
    page_no_last = 137
    version_no = 3

    # 페이지별로 반복하면서 데이터 다운로드
    for pno in range(page_no, page_no_last+1):

        print("===== " + str(pno) + " page start! =====")
        # 한 페이지 당 소요되는 시간 체크
        s_time = time.time()

        # open API 호출
        req = urllib.request.Request("https://api.generated.photos/api/v1/faces?api_key=Your API Key&emotion=neutral&ethnicity=asian&per_page=100&version="
                                     +str(version_no)+"&page="+str(pno))
        data = str(urllib.request.urlopen(req).read())

        # JSON 파싱 테스트
        # f = open("./response_v3_page6.txt", 'r')
        # data = f.read()
        # f.close()

        # print(type(data))
        # print(data)

        # JSON에서 불필요한 문자(b') 삭제하기 (data가 str타입일 경우)
        data = data[2:len(data)-1]
        print(data)

        # JSON 파싱
        jObject = json.loads(str(data))
        faces = jObject.get("faces")

        # 몇 번째 이미지인지
        img_no = 1

        for img in faces:
            img_id = img['id']
            img_url = img['urls'][4]['512']
            gender = img['meta']['gender'][0]
            hair_len = img['meta']['hair_length'][0]

            print(img_url)
            # print(gender)
            # print(hair_len)

            save_dir = 'D:/asian_face2/' + gender + '_' + hair_len + '/' + img_id + ".jpg"

            # 이미지 다운받기
            urllib.request.urlretrieve(img_url, save_dir)

            print("img_no : ", str(img_no), " / img_id : ", str(img_id))
            time.sleep(1)

            img_no = img_no + 1

        print("=== " + str(pno) + " 페이지 걸린 시간 : " + str(time.time()-s_time) + " ===")

    # JSON 파싱 테스트
    # f = open("./response.txt", 'r')
    # data = f.read()
    # f.close()

    # f = open("./response_v2.txt", "w")
    # f.write(str(data))
    # f.close()