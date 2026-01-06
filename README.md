# 단백질의 생물학적 기능 예측 (파일럿)

Kaggle 의 CAFA6 대회(단백질 기능 예측 대회)에 제출할 모델 중 한 파이프라인인 Diamond 파트를 구현<br/>
기간: 2025.12.22 - 2026.01.03<br/>
<br/>
메인 파이프라인: Diamond + LMDB<br/>

- Diamond: BLAST 대비 20,000배 가속화<br/>
- 도메인별 특성 분석 (BP/CC 높음, MF 상대적 낮음)<br/>
- 정밀도 임계값 최적화 (0.7 → 0.5)<br/>

LMDB 데이터 구조 선택<br/>

- Diamond의 헤더 길이 제한 우회<br/>
- Zero-latency GO-Term 조회 (0.1초)<br/>

기술 검증: KNN/FAISS<br/>

- scikit-learn KNN (CPU/RAM 병목) vs FAISS (GPU 가속)<br/>
- 마이그레이션 이슈: 라벨링 호환 불가, 임베딩 재생성 필요. 기존 임베딩 type이 float16으로 되어있어 float32 임베딩 필요.<br/>
- 비용-효율 분석: 마이그레이션 비용 > 성능 이득<br/>
- 최종 결정: Diamond+LMDB에 리소스 집중<br/>

## 폴더구조

```bash
Repository/
├─ .github/workflows/
│    └─ train.yml  ⚠️ 모델 등록 이름 변경 시에만 수정
├─ src/
│    ├─ ensemble.py ✅ ensemble 파이프라인 파일
│    ├─ index.py    ✅ 최초 시작 파일
│    └─ score.py ❌ 수정하면 안 됨 (실시간 엔드포인트 배포 시에만 사용)
├─ train-job.yml  ⚠️ 환경 변경 시에만 수정 가능
└─ README.md
```

## github push 가이드

### Job Push / Register Model

#### 파일명 고정

- 추가 파일 있을 시에 경로는 `src` 내부에서 작업해야 함.

```bash
train.yml
ensemble.py
train-job.yml
index.py
```

#### 저장할 때 디렉토리 설정 `./outputs` 로 고정

- index. py

```python
# ================== 1. config 세팅 ==================
.
.
.

parser.add_argument('--output_dir', type=str, default='./outputs', help='결과 저장 경로')

.
.
.
```

#### config 파라미터 필요하면 추가

- index.py

```python
# ================== 1. config 세팅 ==================
# Azure ML 경로 설정
parser.add_argument('--data_path', type=str, required=True, help='dataset 폴더 경로')
parser.add_argument('--output_dir', type=str, default='./outputs', help='결과 저장 경로')
parser.add_argument('--threads', type=int, default=8)
# === 필요하면 주석해제 후 사용하기
# parser.add_argument('--train_batch_size', type=int, default=1024, help='Head 학습 시 배치 크기 (H5 기반이라 크게 가능)')
# parser.add_argument('--predict_batch_size', type=int, default=2048, help='추론 시 배치 크기')
# parser.add_argument('--alpha', type=float, default=0.6, help='ESM2 가중치')
```

- train-job.yml

```yaml
.
.
.

command: >-
  python input.py
    --data_path ${{inputs.data}}
    --output_dir ${{outputs.result}}
    --batch_size 1 \
    --alpha 0.6 \
    --knn_weight 0.3 \

.
.
.
```

#### 학습 환경 변경 시

- train-job.yml

`azureml:` 뒤의 이름 변경 `@latest`는 마지막 버전을 쓰겠다는 의미. 버전 명시해도 됨.<br /><br />
ex)<br />
diamond_ensenble:1 -> 데이터자산 diamond_ensenble 의 1버전<br />
FOR-CAFA-6 -> 클러스터명<br />
ensemble-env@latest -> cafa6-torch-env 의 마지막 버전<br />
`display_name` 는 선택적으로 추가. 미기입 시, 랜덤 이름 부여. 따옴표 필수.
`experiment_name` 는 선택적으로 추가. 미기입 시, `protein-lab-supporter` 로 고정. 따옴표 필수.

```yaml
.
.
.

inputs:
  data:
    type: uri_folder
    path: azureml:diamond_ensenble@latest
    mode: ro_mount
.
.
.

compute: azureml:github
environment: azureml:ensemble-env@latest
# Job 실험명과 task 명 지정. 미기재 해도 됨.
experiment_name: "실험명"
display_name: "작업명"
```

- train.yml

Job까지만 돌릴 거면 <span style="color:red;">Run Job</span> 만 주석 해제.<br />
<br />
모델 등록까지 돌릴 거면 <span style="color:pink;">Register Model</span> 까지 주석 해제.<br />
`az ml model create --name model` 에서 model 을 변경해도 상관 없음. 이건 azure ml 에 등록되는 이름.<br />
같은 걸 사용하면 version 이 업그레이드 되는 방식.<br />

```yaml
.
.
.

      - name: Run Job
        run: |
          JOB_NAME=$(az ml job create --file azureml/train-job.yml --query name -o tsv)

          echo "JOB_NAME=$JOB_NAME" >> $GITHUB_ENV
          echo "✅ Job 제출 완료!"

      # - name: Register Model
      #   run: |
      #     az ml model create --name model \
      #       --path azureml://jobs/${{ env.JOB_NAME }}/outputs/artifacts/paths/outputs/ \
      #       --type custom_model
```
