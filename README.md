# protein-lab-supporter

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

<br /><br />

### 엔드포인트 배포 (여긴 아직 신경쓸 필요 x)

score.py에서 필요한 model.py 복사

```python
# train. py

# torch.save 바로 아래에 train. py ✅ 추가
model_py_path = Path(__file__).parent / 'model.py'
if model_py_path.exists():
    shutil.copy(str(model_py_path), str(output_dir / 'model.py'))
    print(f"✅ model.py copied to {output_dir / 'model.py'}")
```

#### 가중치 객체 변수 `model_config` 로 고정

```python
# train. py

# ==================== 모델 생성 ====================
print("\n" + "=" * 70)
print("🏗️ 모델 생성")
print("=" * 70)

model_config = ModelConfig(
    embedding_dim=args.embedding_dim,
    num_classes=num_go_terms,
    conv_channels=args.conv_channels,
    kernel_sizes=args.kernel_sizes,
    fc_dims=args.fc_dims,
    dropout=args.dropout,
    conv_dropout_ratio=args.conv_dropout_ratio,
    use_residual=args.use_residual,
    pooling_mode=args.pooling_mode,
    use_batch_norm=True,
    activation='relu'
)
```
