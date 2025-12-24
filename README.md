# protein-lab-supporter

### 폴더구조

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

### github push 가이드

가중치 객체 변수 `model_config` 로 고정

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

저장할때 모델명 `model.pth` 로 고정

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
