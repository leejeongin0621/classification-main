## How to use
utils/Model.py에서 model 구현

MODEL_MAKER.py에서 model 설정 후 실행 -> model.pt (저장 모델 이름) 생성

-> model 선언 및 저장 이름 확실히 설정

IMU_main.py에서 model.pt (저장 모델 이름) 지정해서 학습

## 정인이 할 일
### Due date: Mar 06
TODO: utils/Model.py (IMU_BiLSTM Code만), MODEL_MAKER.py, IMU_main.py 코드 돌려보면서, pytorch 기반 DL system 공부
+ Code의 각 파트가 어떤 역할인지 주석으로 코드에 정리 (각 줄마다 적을 필요는 없음)
+ 주석 적을 때 되도록이면 영어로 작성 (한글로 쓰면 나중에 코드 열었다가 깨지는 경우 존재)

Model.py에서, 필요한 부분 (IMU_BiLSTM) 파트 외에는 우선 주석으로 처리해둠, 필요하면 해당 부분도 공부해봐

기존 코드들 가져와서 정리한거라, 일부 legacy code가 있을 수 있는데, 그건 조금만 무시 부탁... (구분 어려우면 언제든 질문줘)

### Future Work
GNN, SNN 등 여러가지 기법 사용해서 모델 구현 및 최적화, XAI 등의 방법을 졸작 주제로 설정 예정

상세한 주제는 추후 결정해줄거니, 이에 대비해서 model part를 조금 더 중점적으로 공부하면 좋을듯

