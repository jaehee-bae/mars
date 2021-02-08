# 얼굴 만들기
얼굴부위(눈, 눈썹, 코, 입)를 조합 후 채색하여 새로운 얼굴을 만드는 방법과 필요한 코드에 대해서 설명한다.

##1. 데이터셋 준비
1) 얼굴이미지 준비
<br>generated photo 데이터셋을 사용하며, 사용할 수 있는 방법은 두 가지이다.

    1) Bulk로 다운받은 데이터셋에서 정면얼굴 이미지만 골라서 사용
        + 다운받은 데이터셋은 MARS 구글 드라이브에 있다.
        + 이미지 사이즈는 1024 x 1024
    2) open API로 원하는 특징의 이미지만 골라서 사용
        + generated photos 사이트에서 회원가입 후, 발급된 API key가 필요하다.
        + 한 달에 50번만 무료로 호출이 가능하며, 한 번 호출할 때마다 100개까지 다운로드가 가능함
        + 이미지 사이즈는 512 x 512
        + download_generated_photos.py 코드에서 발급받은 API key 입력 후, 실행하면 다운받을 수 있다.
        + <code>python download_generated_photos.py</code>

2) 얼굴부위 이미지 준비
<br>1번에서 구한 얼굴이미지를 facial landmark를 활용하여 얼굴 부위 별로 크롭한다.
<br> <code>python crop_features_main.py</code>

##2. 얼굴 만들기
얼굴을 만드는 방법은 다음 4단계로 진행된다.
<img src="/assemble_face.png"></img>