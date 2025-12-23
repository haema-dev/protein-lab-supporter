# protein-lab-supporter

폴더구조

```bash
Repository/
├─ .github/workflows/
│ ├─ train.yml  ❌ 학습자가 수정하면 안 됨
│ └─ deploy.yml ❌ 학습자가 수정하면 안 됨
├─ src/
│ ├─ train.py ✅ 작업할 곳
│ └─ score.py ❌ 학습자가 수정하면 안 됨
├─ azureml/
│ ├─ train-job.yml  ⚠️ display_name 만 수정 가능 (Job 이름 설정. 비워둬도 됨)
│ ├─ deploy.yml     ❌ 학습자가 수정하면 안 됨
│ └─ endpoint.yml   ❌ 학습자가 수정하면 안 됨
└─ README.md
```
