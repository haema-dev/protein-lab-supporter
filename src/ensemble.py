import os
import h5py
import pickle
import subprocess
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import re
import lmdb
from tqdm import tqdm
from Bio import SeqIO
from loguru import logger
from torch.utils.data import Dataset, DataLoader
import json

# [유틸리티]
def clean_id(full_id):
    if '|' in full_id: return full_id.split('|')[1]
    return full_id

def is_valid_go_term(term):
    return bool(re.match(r"^GO:\d{7}$", str(term).strip()))

class DiamondESM2Processor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = config['output_dir']
        self.model_path = os.path.join(self.output_dir, "models", "head_model.pt")

    def load_go_mapping(self, tsv_path):
        df = pd.read_csv(tsv_path, sep='\t')
        # ✅ 'protein_id' → 'EntryID', 'go_id' → 'term'
        return df.groupby('EntryID')['term'].apply(lambda x: list(set(x))).to_dict()

    def generate_label_list(self, tsv_path, output_path):
        df = pd.read_csv(tsv_path, sep='\t')
        # ✅ 'go_id' → 'term'
        all_labels = sorted(df['term'].unique().tolist())
        with open(output_path, 'wb') as f: 
            pickle.dump(all_labels, f)
        return len(all_labels)

    def clean_id(self, header):
        """
        [수정됨] header 문자열에서 ID를 추출합니다.
        self는 클래스 인스턴스 자체이므로, 'in' 연산은 header에 수행해야 합니다.
        """
        # header가 문자열인지 확인 (안전장치)
        header = str(header) 
        
        if "|" in header:
            # sp|ID|NAME 형식인 경우 두 번째 요소 반환
            return header.split("|")[1]
        
        # 공백이 있는 경우 첫 번째 단어만 반환
        return header.split()[0]

    def build_diamond_lmdb(self, fasta_in, go_mapping, lmdb_path, db_out, pkl_path=None, npz_path=None):
        # 1. 조상 정보 로드 로직 (기존과 동일)
        go_map_data = go_mapping
        ancestor_map = {}
        if pkl_path and npz_path and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f: 
                go_ids = pickle.load(f)
            matrix = sp.load_npz(npz_path)
            r, c = matrix.nonzero()
            for row, col in zip(r, c):
                if row != col:
                    child, parent = go_ids[row], go_ids[col]
                    if child not in ancestor_map: ancestor_map[child] = set()
                    ancestor_map[child].add(parent)

        # 2. LMDB 구축
        env = lmdb.open(lmdb_path, map_size=20 * 1024**3)
        with env.begin(write=True) as txn:
            for record in tqdm(SeqIO.parse(fasta_in, "fasta"), desc="📦 LMDB(JSON) 구축"):
                # [중요] acc_id가 'A0A0C5B5G6'가 됩니다.
                acc_id = self.clean_id(record.id) 
                
                desc = record.description
                org_name = "Unknown"
                org_id = "0"
                
                if "OS=" in desc:
                    os_part = desc.split("OS=")[1]
                    # "Homo sapiens" 추출 (strain 정보 제외)
                    full_org_name = os_part.split(" OX=")[0].strip()
                    org_name = full_org_name.split(" (")[0].strip()
                    
                    # "9606" 추출
                    if "OX=" in os_part:
                        org_id = os_part.split("OX=")[1].split()[0].strip()

                # GO Term 매칭 및 확장
                terms = set(go_map_data.get(acc_id, []))
                expanded = terms.copy()
                for t in terms: 
                    expanded.update(ancestor_map.get(t, []))
                
                # [사용자 요청 반영] JSON 데이터 구조
                data_dict = {
                    "tax_id": acc_id,        # 질문에서 말씀하신 대로 단백질 ID(A0A0C5B5G6)를 넣음
                    "org_id": org_id,        # 9606 (Taxonomy ID)
                    "org_name": org_name,    # Homo sapiens (종 이름)
                    "go_terms": sorted(list(expanded))
                }
                
                # JSON 직렬화
                json_value = json.dumps(data_dict, ensure_ascii=False)
                
                # [핵심] LMDB의 Key를 'A0A0C5B5G6'로 저장합니다.
                txn.put(acc_id.encode('utf-8'), json_value.encode('utf-8'))
                
        env.close()
        
        # 3. DIAMOND DB 생성 (config의 스레드 값 적용)
        num_threads = self.config.get('threads', 4)
        subprocess.run(["diamond", "makedb", "--in", fasta_in, "-d", db_out, "-p", str(num_threads)], check=True)

    def run_diamond_search(self, query_fasta, db_path, result_tsv):
        logger.info(f"💎 Diamond Search (Threads: {self.config['threads']})")
        cmd = ["diamond", "blastp", "-q", query_fasta, "-d", db_path, "-o", result_tsv, 
               "-p", str(self.config['threads']), "--max-target-seqs", "1", "--outfmt", "6"]
        subprocess.run(cmd, check=True)

    def final_ensemble(self, result_hits, lmdb_path, esm_preds=None, label_list_path=None):
        """
        앙상블용 고정밀도 Diamond 컴포넌트
        """
        import json
        import pandas as pd
        from tqdm import tqdm
        import lmdb
        
        columns = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 
                'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
        
        try:
            dmnd_df = pd.read_csv(result_hits, sep='\t', names=columns)
            initial_count = len(dmnd_df)
            
            # ✅ 앙상블용 고정밀도 필터
            pident_threshold = 50
            dmnd_df = dmnd_df[
                (dmnd_df['pident'] >= pident_threshold) &  # 조정 가능 (40, 50, 60)
                (dmnd_df['evalue'] <= 1e-10) &             # 극도로 엄격
                (dmnd_df['bitscore'] >= 100) &             # 높은 품질
                (dmnd_df['length'] >= 80)                  # 긴 alignment
            ]
            
            logger.info(f"High-precision filtering (pident≥{pident_threshold}): "
                    f"{initial_count} -> {len(dmnd_df)} hits ({len(dmnd_df)/initial_count*100:.1f}%)")
            
            dmnd_dict = {k: v for k, v in dmnd_df.groupby('qseqid')}
            
        except Exception as e:
            logger.warning(f"⚠️ Diamond 결과 로드/필터링 실패: {e}")
            return pd.DataFrame(columns=['Protein Id', 'GO Term Id', 'Prediction'])
        
        # LMDB 조회
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        final_subs = []
        
        with env.begin() as txn:
            for qid, hits in tqdm(dmnd_dict.items(), desc="High-Precision Diamond"):
                combined_scores = {}
                
                for _, row in hits.iterrows():
                    sseqid = self.clean_id(row['sseqid'])
                    val = txn.get(sseqid.encode('utf-8'))
                    
                    if val:
                        data = json.loads(val.decode('utf-8'))
                        go_list = data.get('go_terms', [])
                        
                        if go_list:
                            # ✅ 높은 pident는 높은 신뢰도
                            confidence = row['pident'] / 100.0
                            
                            for go_id in go_list:
                                combined_scores[go_id] = max(
                                    combined_scores.get(go_id, 0), 
                                    confidence
                                )
                
                # ✅ 높은 threshold (고정밀도)
                for go_id, s in combined_scores.items():
                    if s >= 0.40:  # 40% 이상 (매우 엄격)
                        final_subs.append([qid, go_id, round(s, 3)])
        
        env.close()
        logger.info(f"High-precision predictions: {len(final_subs)}")
        return pd.DataFrame(final_subs, columns=['Protein Id', 'GO Term Id', 'Prediction'])

    def create_cafa_submission(self, df, team_name, model_num):
        out_file = os.path.join(self.output_dir, f"submission_{model_num}.tsv")
        with open(out_file, 'w') as f:
            f.write(f"AUTHOR {team_name}\nMODEL {model_num}\nKEYWORDS Diamond-LMDB\n")
            df.to_csv(f, sep='\t', index=False, header=False)
        return out_file
    
    def evaluate_diamond_only(self, result_tsv, lmdb_path, label_list_path):
        """Diamond BLAST 결과만으로 평가 - JSON 대응 버전"""
        logger.info("📊 [Ablation] Diamond-only evaluation (JSON parsing)...")
        
        try:
            # 결과 로드
            dmnd_df = pd.read_csv(result_tsv, sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
            dmnd_dict = {k: v for k, v in dmnd_df.groupby('qseqid')}
            
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            diamond_subs = []
            
            with env.begin() as txn:
                for qid, hits in tqdm(dmnd_dict.items(), desc="🔍 Analyzing Hits"):
                    comb = {}
                    for _, row in hits.iterrows():
                        # ✅ 검색 결과 ID도 clean_id로 정제해서 조회
                        sseqid = self.clean_id(row['sseqid'])
                        val = txn.get(sseqid.encode('utf-8'))
                        
                        if val:
                            # ✅ 핵심 수정: JSON 로드
                            data = json.loads(val.decode('utf-8'))
                            go_list = data.get('go_terms', [])
                            
                            score = row['pident'] / 100.0
                            for go_id in go_list:
                                if is_valid_go_term(go_id):
                                    comb[go_id] = max(comb.get(go_id, 0), score)
                    
                    for go_id, f_score in comb.items():
                        diamond_subs.append([qid, go_id, round(f_score, 3)])
            
            env.close()
            diamond_df = pd.DataFrame(diamond_subs, columns=['Protein Id', 'GO Term Id', 'Prediction'])
            output_file = os.path.join(self.config['output_dir'], "diamond_only_submission.tsv")
            diamond_df.to_csv(output_file, sep='\t', index=False)
            
            logger.success(f"✅ Diamond-only predictions: {len(diamond_subs)}")
            return diamond_df
            
        except Exception as e:
            logger.error(f"❌ Diamond-only evaluation failed: {e}")
            raise