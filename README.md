# protein-lab-supporter

## 폴더구조

```bash
Repository/
├─ .github/workflows/
│    └─ train.yml  ❌ 학습자가 수정하면 안 됨
├─ src/
│    ├─ train.py ✅ 작업할 곳
│    ├─ model.py ✅ 작업할 곳
│    └─ score.py ❌ 학습자가 수정하면 안 됨
├─ azureml/
│    └─ train-job.yml  ⚠️ display_name 만 수정 가능 (Job 이름 설정. 비워둬도 됨)
└─ README.md
```

## github push 가이드

### 모델 자동 등록까지

#### 파일명 고정

```bash
train.py
model.py
model.pth
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

#### 저장할 때 디렉토리 설정 `./outputs` 로 고정

```python
# train. py

# ==================== 출력 ====================
parser.add_argument('--output_dir', type=str, default='./outputs',
                    help='모델 저장 디렉토리')
parser.add_argument('--log_interval', type=int, default=10,
                    help='로그 출력 주기 (배치 단위)')
```

#### 저장할때 모델명 `model.pth` 로 고정

```python
# train. py

# Checkpoint 저장
checkpoint_path = output_dir / 'model.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'model_config': model_config.to_dict(),
    'training_config': training_config.to_dict(),
    'args': vars(args),
}, checkpoint_path)
```

#### 학습 환경 변경 시

`azureml:` 뒤의 이름 변경 `@latest`는 마지막 버전을 쓰겠다는 의미. 버전 명시해도 됨.<br /><br />
ex)<br />
cafa_6:1 -> 데이터자산 cafa_6 의 1버전<br />
FOR-CAFA-6 -> 클러스터명<br />
cafa6-torch-env@latest -> cafa6-torch-env 의 마지막 버전<br />
`display_name` 는 선택적으로 추가. 미기입 시, 랜덤 이름 부여. 따옴표 필수.

```yaml
# train-job.yml

.
.
.

inputs:
  cafa_data:
    type: uri_folder
    path: azureml:cafa_6@latest
    mode: ro_mount

.
.
.


compute: azureml:FOR-CAFA-6
environment: azureml:cafa6-torch-env@latest
display_name: "이름"
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
