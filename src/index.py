import os
import argparse
import subprocess
import torch
import pandas as pd
import pickle
from loguru import logger
import traceback
import glob
import math

# [1] 모듈 로드 (기존 ensemble.py에 DiamondESM2Processor가 있다고 가정)
from ensemble import DiamondESM2Processor


def convert_size(size_bytes):
    """
    파일 용량 체크
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def find_file(directory, pattern, required=False):
    """
    개선된 파일 찾기 함수: 
    1. 확장자만 넣어도 자동으로 '*'를 붙여 검색합니다.
    2. 키워드 매칭을 대소문자 구분 없이 수행합니다.
    """
    from pathlib import Path
    
    if not os.path.isdir(directory):
        if required:
            logger.error(f"❌ 디렉토리 없음: {os.path.abspath(directory)}")
            raise FileNotFoundError(f"디렉토리 없음: {directory}")
        return None
    
    # 패턴 보정: ".fasta" -> "*.fasta"
    search_pattern = pattern
    if not search_pattern.startswith("*"):
        search_pattern = f"*{search_pattern}"
    
    candidates = list(Path(directory).glob(search_pattern))
        
    found_path = str(candidates[0])
    logger.info(f"🔍 파일 발견: {os.path.basename(found_path)}")
    return found_path

def check_model_sizes(directory, extension="*.pt"):
    """
    특정 폴더 내 모델 파일들만 체크
    """
    files = glob.glob(os.path.join(directory, extension))
    
    logger.info(f"Checking {len(files)} files in {directory}...")
    
    total_size = 0
    
    for file in files:
        size = os.path.getsize(file)
        total_size += size
        logger.info(f"File: {os.path.basename(file)} | Size: {convert_size(size)}")
        
    logger.success(f"Total Size: {convert_size(total_size)}")

def main():
    parser = argparse.ArgumentParser(description="DiamondDB + LMDB")
    
    # ================== 1. config 세팅 ==================
    # Azure ML 경로 설정
    parser.add_argument('--data_path', type=str, required=True, help='dataset 폴더 경로')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='결과 저장 경로')
    parser.add_argument('--threads', type=int, default=14)
    parser.add_argument('--fs_score', type=float, default=0.99)
    parser.add_argument('--pident', type=int, default=50)
    parser.add_argument('--evalue', type=float, default=1e-5)
    # === 필요하면 주석해제 후 사용하기
    # parser.add_argument('--train_batch_size', type=int, default=1024, help='Head 학습 시 배치 크기 (H5 기반이라 크게 가능)')
    # parser.add_argument('--predict_batch_size', type=int, default=2048, help='추론 시 배치 크기')

    args = parser.parse_args()
    
    # ================== 1. 경로 및 환경 설정 ==================
    # Root DIR 설정
    DATASET_DIR = args.data_path
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"📂 Input/Output 디렉토리 설정 완료:")
    logger.info(f"   - Input Dataset Root: {os.path.abspath(DATASET_DIR)}")
    logger.info(f"   - Output Root: {os.path.abspath(OUTPUT_DIR)}")

    # 모델 저장 디렉토리 설정 및 생성
    # Azure ML에서는 ./outputs 폴더 안에 파일을 두면 자동으로 아티팩트로 수집
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # (옵션) 로그나 임시 파일을 위한 폴더
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.info(f"📂 작업 디렉토리 설정 완료:")
    logger.info(f"   - Model Save Dir: {os.path.abspath(MODEL_DIR)}")
    logger.info(f"   - Log Save Dir: {os.path.abspath(LOG_DIR)}")

    # 1. 디렉토리 설정
    # [ Input dataset folder structure ]
    # ─┬─ fasta (.fasta) :  large_learning_superset Data
    #  ├─ foldseek (.tsv)
    #  ├─ h5 (.h5) :  esm2-3b float16 임베딩 데이터
    #  ├─ interpro (.tsv)
    #  ├─ ontology (.npz / .pkl) :  부모전파 파일
    #  ├─ tsv (.tsv) :  large_learning_superset Data
    #  └─ validation :  채점 데이터
    FASTA_DIR = os.path.join(DATASET_DIR, "fasta")
    FOLDSEEK_DIR = os.path.join(DATASET_DIR, "foldseek")
    TSV_DIR = os.path.join(DATASET_DIR, "tsv")
    H5_DIR = os.path.join(DATASET_DIR, "h5")
    INTERPRO_DIR = os.path.join(DATASET_DIR, "interpro")
    ONTOLOGY_DIR = os.path.join(DATASET_DIR, "ontology")
    VALID_DIR = os.path.join(DATASET_DIR, "validation")

    # 2. 동적 파일 찾기 (이름이 달라도 확장자와 키워드로 자동 매칭)
    # Train 데이터
    TRAIN_FASTA = find_file(FASTA_DIR, ".fasta")
    TRAIN_GO_TSV = find_file(TSV_DIR, ".tsv")
    TRAIN_H5    = find_file(H5_DIR, ".h5")

    # Ontology (NPZ, PKL은 보통 하나뿐이므로 확장자로 검색)
    PARENTS_PKL = find_file(ONTOLOGY_DIR, ".pkl")
    PARENTS_NPZ = find_file(ONTOLOGY_DIR, ".npz")

    # Test / Validation 데이터
    TEST_FASTA    = find_file(VALID_DIR, ".fasta") or find_file(VALID_DIR, ".fasta")
    TEST_H5       = find_file(VALID_DIR, ".h5") or find_file(VALID_DIR, ".h5")
    VALIDATION_GT = find_file(VALID_DIR, "*.tsv")

    # 경로 검증 로그
    logger.info("🔍 동적 경로 탐색 결과:")
    logger.info(f"  - Train FASTA: {TRAIN_FASTA}")
    logger.info(f"  - Train H5: {TRAIN_H5}")
    logger.info(f"  - Train ONTOLOGY: {ONTOLOGY_DIR}")
    logger.info(f"  - Test VALID: {VALID_DIR}")

    logger.info(f"🚀 DiamondDB + LMDB 파이프라인 시작 (Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # ================== 2. 프로세서 초기화 ==================
    config = {
        'threads': args.threads,
        'output_dir': OUTPUT_DIR,
        'fs_score': args.fs_score,
        'pident': args.pident,
        'evalue': args.evalue,
        # 'train_batch_size': args.train_batch_size,
        # 'batch_size': args.predict_batch_size, 
        # 'embedding_dim': 2560 ### 수정하면 안 됨.
    }

    proc = DiamondESM2Processor(config)
    
    # ================== 3. [Phase 1] DiamondDB + LMDB 매핑저장 ==================
    try:
        # Step 1: GO Mapping 로드
        go_mapping = proc.load_go_mapping(TRAIN_GO_TSV)
        logger.success(f"✅ {len(go_mapping)}개 단백질-GO 매핑 로드!")
        
        # Step 2: Label List 생성 (go_mapping에서 추출)
        label_pkl = os.path.join(MODEL_DIR, "labels.pkl")
        proc.generate_label_list(TRAIN_GO_TSV, label_pkl)
        # 파일 확인 및 용량 체크
        if os.path.exists(label_pkl):
            file_size = os.path.getsize(label_pkl)
            logger.success(f"✅ labels.pkl 생성 성공!")
            logger.info(f"📊 파일 위치: {os.path.abspath(label_pkl)}")
            logger.info(f"💾 모델(레이블) 파일 크기: {convert_size(file_size)}")
            # === 필요하면 주석 해제 ===
            # [검증] 실제 데이터 로드 테스트
            # with open(label_pkl, 'rb') as f:
            #     loaded_labels = pickle.load(f)
            #     logger.info(f"🔢 총 GO Term 개수: {len(loaded_labels):,}개")
        else:
            logger.error(f"❌ labels.pkl 파일이 생성되지 않았습니다! 경로를 확인하세요: {label_pkl}")
        
        # Step 3: LMDB 구축 (go_mapping 활용)
        lmdb_path = os.path.join(MODEL_DIR, "train_lmdb")
        dmnd_db = os.path.join(MODEL_DIR, "diamond_db.dmnd")
        proc.build_diamond_lmdb(
            TRAIN_FASTA,
            go_mapping,
            lmdb_path,
            dmnd_db,
            PARENTS_PKL,
            PARENTS_NPZ
        )
        logger.success("✅ DiamondDB + LMDB 매핑저장 성공!")

        # FS_DB = os.path.join(MODEL_DIR, "foldseek")
        # logger.success("✅ DiamondDB + LMDB 매핑저장 성공!")
        
    except Exception as e:
        logger.error(f"❌ DiamondDB + LMDB 매핑저장 중 오류 발생: {e}")
        return

    # ================== 4. [Phase 2] 학습 (Training) ==================
    logger.info("🏗️ Phase 1: 데이터 로드 및 학습 시작")
    
    ### 학습 로직 수행
    
    logger.info("⏩ 학습 단계 건너뛰기")

    # ================== 4. [Phase 3] 추론 (Inference) ==================
    logger.info("🚀 Phase 2: 추론 시작")
    try:
        # 1. Diamond 검색
        dmnd_hits = os.path.join(OUTPUT_DIR, "dmnd_hits.tsv")
        proc.run_diamond_search(TEST_FASTA, dmnd_db, dmnd_hits)
        
        esm_preds = None
        
        ### 2. 학습한 모델추가
        
        logger.info("학습한 모델의 예측 수행: {}", esm_preds)
        
        # 3. 최종 앙상블
        INTERPRO_FILE = find_file(INTERPRO_DIR, ".tsv")
        FOLDSEEK_FILE = find_file(FOLDSEEK_DIR, ".tsv")

        final_df = proc.final_ensemble(
            dmnd_hits=dmnd_hits,
            lmdb_path=lmdb_path,
            interpro_path=INTERPRO_FILE,
            submission_path=FOLDSEEK_FILE
        )

        
        # 4. 결과 저장
        final_save_path = os.path.join(OUTPUT_DIR, "final_results.tsv")
        final_df.to_csv(final_save_path, sep='\t', index=False)
        logger.success(f"✅ 추론 완료! 결과 저장됨: {final_save_path}")

    except Exception as e:
        logger.error(f"❌ 추론 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return

    # ================== 5. [Phase 4] Ablation Study ==================
    if VALIDATION_GT is not None and os.path.exists(VALIDATION_GT):
        logger.info("🔬 Phase 3: Ablation Study & Evaluation")
        try:

            ### 필요하면 추가하면 됨

            logger.success("✅ 평가 완료!")
        except Exception as e:
            logger.warning(f"⚠️ 평가 단계 실패: {e}")
    else:
        logger.warning("⚠️ 파일을 찾을 수 없습니다. 평가 스킵")

    logger.success("🏁 CAFA6 통합 파이프라인 종료!")

if __name__ == "__main__":
    main()