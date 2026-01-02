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
from collections import defaultdict

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
        self.threads = config.get("threads", 14)        # default 14
        self.fs_score = config.get("fs_score", 0.99)    # default 0.99
        self.pident = config.get("pident", 50)          # default 50
        self.evalue = config.get("evalue", 1e-5)        # default 1e-5

    def load_go_mapping(self, tsv_path):
        """GO ID별 Namespace 정보를 함께 저장하도록 수정"""
        logger.info(f"📂 GO 매핑 로드 중: {tsv_path}")
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Protein ID -> GO terms 매핑
        mapping = df.groupby('EntryID')['term'].apply(lambda x: list(set(x))).to_dict()
        
        # ✅ 핵심: GO Term -> Namespace 매핑 저장
        # 컬럼명이 'namespace' 또는 'aspect'인지 확인하세요.
        if 'namespace' in df.columns:
            # 🔍 로깅 추가 시작
            unique_ns = df['namespace'].unique()
            logger.info(f"🔍 Unique namespaces found: {unique_ns}")
            logger.info(f"📊 Total GO terms: {len(df['term'].unique())}")
            # 🔍 로깅 추가 끝
            self.go_info_dict = pd.Series(df.namespace.values, index=df.term).to_dict()
        elif 'aspect' in df.columns:
            # 🔍 로깅 추가 시작
            unique_ns = df['aspect'].unique()
            logger.info(f"🔍 Unique aspects found: {unique_ns}")
            logger.info(f"📊 Total GO terms: {len(df['term'].unique())}")
            # 🔍 로깅 추가 끝
            self.go_info_dict = pd.Series(df.aspect.values, index=df.term).to_dict()
        else:
            raise ValueError("❌ 'namespace' 또는 'aspect' 컬럼이 없습니다!")
        
        return mapping

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

                ### 부모전파 O
                expanded = terms.copy()
                for t in terms: 
                    expanded.update(ancestor_map.get(t, []))
                data_dict = {
                    "protein_id": acc_id,    # 질문에서 말씀하신 대로 단백질 ID(A0A0C5B5G6)를 넣음
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

    def final_ensemble(self, dmnd_hits, lmdb_path, interpro_path=None, submission_path=None):
        if not hasattr(self, 'go_info_dict') or self.go_info_dict is None:
            raise ValueError("❌ go_info_dict가 없습니다!")
        
        mf_terms = {term for term, ns in self.go_info_dict.items() 
                    if ns in ['MFO', 'molecular_function', 'F']}
        logger.info(f"✅ MF Terms: {len(mf_terms)}")
        # 1. Diamond 점수 산출 (MF 성능 보존의 핵심)
        combined_dict = defaultdict(lambda: defaultdict(float))
        try:
            dmnd_df = pd.read_csv(dmnd_hits, sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
            dmnd_df = dmnd_df[(dmnd_df['pident'] >= self.pident) & (dmnd_df['evalue'] <= self.evalue)]

            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            with env.begin() as txn:
                for qid, group in dmnd_df.groupby('qseqid'):
                    term_ev = defaultdict(list)
                    for _, row in group.iterrows():
                        val = txn.get(self.clean_id(row['sseqid']).encode('utf-8'))
                        if val:
                            go_list = json.loads(val.decode('utf-8')).get('go_terms', [])
                            conf = float(row['pident'] / 100.0)
                            for t in go_list: term_ev[t].append(conf)
                    for t, evs in term_ev.items():
                        combined_dict[qid][t] = float(1.0 - np.prod([1.0 - e for e in evs]))
            env.close()
        except Exception as e:
            logger.warning(f"Diamond 처리 중 알림: {e}")

        df_diamond = pd.DataFrame([[q, t, s] for q, t_dict in combined_dict.items() for t, s in t_dict.items()], columns=['id', 'term', 'score_dmnd'])

        # ✅ MF Term 분리
        mf_terms = {term for term, ns in self.go_info_dict.items() 
                    if ns in ['MFO', 'molecular_function', 'F']}

        # ✅ 헬퍼 함수
        def load_and_filter(path, col_name):
            if path and os.path.exists(str(path)):
                df = pd.read_csv(path, sep='\t', header=None, names=['id', 'term', col_name])
                if len(df) > 0:
                    logger.info(f"📊 {col_name} score range: {df[col_name].min():.3f} ~ {df[col_name].max():.3f}")
                df = df[~df['term'].isin(mf_terms)]
                return df[df[col_name] >= 0.6]
            return pd.DataFrame(columns=['id', 'term', col_name])

        # ✅ 데이터 로드
        df_base = load_and_filter(interpro_path, 'score_base')
        df_patch = load_and_filter(submission_path, 'score_patch')

        # ✅ 병합 (None 체크)
        if df_base.empty:
            merged = df_patch.copy() if not df_patch.empty else pd.DataFrame(columns=['id', 'term'])
        else:
            merged = pd.merge(df_base, df_patch, on=['id', 'term'], how='outer')
        merged = pd.merge(merged, df_diamond, on=['id', 'term'], how='outer')

        # ✅ MAX 기반 앙상블
        score_cols = ['score_dmnd', 'score_patch', 'score_base']
        merged['final'] = merged[score_cols].max(axis=1, skipna=True)

        output = merged[['id', 'term', 'final']].dropna(subset=['final']).copy()
        output['final'] = output['final'].round(3)

        logger.success(f"✅ 앙상블 완료: MF(Diamond) {len(output[output['term'].isin(mf_terms)])} + BP/CC {len(output[~output['term'].isin(mf_terms)])}")
        return output
    