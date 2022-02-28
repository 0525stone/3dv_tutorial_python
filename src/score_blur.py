"""
Blur detection

Memo from notes
- 문제 : 번호판 부분이 blur 되어 있는 것에 대한 지표를 뽑아야함
    - Canny는 blur 되어 있는 번호판 부분에 대해서 edge를 하나도 잡지 않음
    - Canny 결과의 variance를 확인해 본 결과, edge가 없는 것과 있는 경우에 대해서 값 차이가 크게 남(그렇기 때문에 배경에서 edge 많이 잡히면 오동작 위험)
- 해결 : 입력 영상에서 차량 번호판이 있을 만한 위치에 대해서 crop을 하고, crop한 영상에 대해서 canny 결과의 variance를 확인 후, 임계치와 비교하여 blur 여부 결정
  - Canny, Laplacian, Sobel 중에서 Canny가 제일 동작을 잘하는 것 확인
    - Laplacian : 2차 미분 필터
    - Sobel : 1차 미분 필터
    - Canny : sobel filter + hyteresis thresholding 으로 에지 여부 판단 -> 결과 제일 명확하게 나옴
"""


