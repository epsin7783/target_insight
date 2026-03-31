"""
K-Means 기반 RFM 고객 군집화 서비스
- scikit-learn KMeans 사용
- Rule-based 추천 엔진 포함
"""
import json
import io
from datetime import date

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── 룰베이스 추천 매핑 ──────────────────────────────────────────────
CLUSTER_RULES = {
    'vip': {
        'label': 'vip',
        'name': 'VIP 고객군',
        'description': '고액·다빈도 결제자로 충성도가 높은 핵심 고객',
        'badge_color': '#6C63FF',
        'channel': '인스타그램 타겟 광고 · VIP 전용 뉴스레터',
        'keywords': '프리미엄, 한정판, VIP 혜택, 시즌 오퍼',
        'message': '최고의 고객님께 드리는 VIP 전용 혜택을 만나보세요.',
    },
    'churn_risk': {
        'label': 'churn_risk',
        'name': '이탈 위험군',
        'description': '최근 방문이 뜸해지고 있는 재유입 필요 고객',
        'badge_color': '#FF6B6B',
        'channel': '네이버 GFA 재방문 유도 · SMS 발송',
        'keywords': '복귀 혜택, 보고싶었습니다, 재방문 쿠폰, 특별 할인',
        'message': '오랜만이에요! 돌아오시면 특별한 혜택을 드릴게요.',
    },
    'potential': {
        'label': 'potential',
        'name': '잠재 고객군',
        'description': '방문은 잦지만 아직 결제 전환이 낮은 육성 대상',
        'badge_color': '#43C6AC',
        'channel': '구글 디스플레이 광고 · 첫 결제 할인 이벤트',
        'keywords': '가성비, 입문 추천, 첫 구매 혜택, 베스트셀러',
        'message': '첫 구매 고객님께 특별 할인 혜택을 드립니다.',
    },
    'general': {
        'label': 'general',
        'name': '일반 고객군',
        'description': '평균적인 구매 패턴을 보이는 일반 고객',
        'badge_color': '#F7B731',
        'channel': '카카오 친구톡 · 페이스북 광고',
        'keywords': '신상품, 이번 주 추천, 멤버십 혜택',
        'message': '신상품과 다양한 혜택을 확인해 보세요.',
    },
}


def _assign_cluster_type(centers: np.ndarray) -> list[str]:
    """
    군집 중심값(R, F, M)을 분석하여 각 군집에 레이블을 부여합니다.
    - R(Recency)은 낮을수록 최근 방문 → 정규화 후 낮은 값이 좋음
    - F(Frequency)는 높을수록 좋음
    - M(Monetary)는 높을수록 좋음
    """
    n = len(centers)
    labels = [''] * n

    # RFM 스코어: -R + F + M (표준화된 값 기준)
    scores = -centers[:, 0] + centers[:, 1] + centers[:, 2]
    rank = np.argsort(scores)[::-1]  # 높은 순

    if n == 3:
        type_map = ['vip', 'churn_risk', 'potential']
    else:
        type_map = ['vip', 'potential', 'general', 'churn_risk']

    for i, idx in enumerate(rank):
        labels[idx] = type_map[i]

    return labels


def run_rfm_clustering(csv_file, n_clusters: int = 3) -> dict:
    """
    CSV 파일을 읽어 RFM K-Means 군집화를 수행하고 결과를 반환합니다.

    Args:
        csv_file: 업로드된 파일 객체 (InMemoryUploadedFile 등)
        n_clusters: 군집 수 (3 또는 4)

    Returns:
        {
          'total': int,
          'n_clusters': int,
          'clusters': [ { label, name, count, avg_r, avg_f, avg_m,
                          channel, keywords, message, scatter } ]
        }
    """
    # ── 1. CSV 파싱 ───────────────────────────────────────────────
    raw = csv_file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), encoding='cp949')

    df.columns = df.columns.str.strip()

    # 컬럼명 유연하게 매핑
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(' ', '')
        if '고객명' in col or 'name' in cl or '이름' in col:
            col_map['name'] = col
        elif any(k in cl for k in ['마지막방문', 'lastvisit', '방문일', '방문날', '최근방문', '날짜', 'date', 'recency', 'visit']):
            col_map['last_visit'] = col
        elif any(k in cl for k in ['구매횟수', 'frequency', '빈도', '횟수', '구매수', '주문수', 'count']):
            col_map['frequency'] = col
        elif any(k in cl for k in ['결제', 'monetary', '금액', '매출', '구매금액', '총액', 'amount', 'revenue', 'sales']):
            col_map['monetary'] = col

    required = ['last_visit', 'frequency', 'monetary']
    missing = [r for r in required if r not in col_map]
    if missing:
        found_cols = ', '.join(f'"{c}"' for c in df.columns)
        label_map = {'last_visit': '방문일', 'frequency': '구매 횟수', 'monetary': '결제 금액'}
        missing_labels = ', '.join(label_map[r] for r in missing)
        raise ValueError(
            f"CSV에서 [{missing_labels}] 컬럼을 인식하지 못했습니다. "
            f"업로드한 파일의 컬럼: {found_cols} — "
            f"샘플 CSV를 다운로드해서 형식을 확인해 주세요."
        )

    # ── 2. Recency 계산 (오늘 기준 경과일) ────────────────────────
    today = pd.Timestamp(date.today())
    df['_last_visit'] = pd.to_datetime(df[col_map['last_visit']], errors='coerce')
    df['R'] = (today - df['_last_visit']).dt.days.fillna(9999).astype(float)
    df['F'] = pd.to_numeric(df[col_map['frequency']], errors='coerce').fillna(0)
    df['M'] = pd.to_numeric(df[col_map['monetary']], errors='coerce').fillna(0)

    if 'name' in col_map:
        df['_name'] = df[col_map['name']].fillna('고객')
    else:
        df['_name'] = [f'고객{i+1}' for i in range(len(df))]

    rfm = df[['R', 'F', 'M']].values

    if len(rfm) < n_clusters:
        raise ValueError(f"데이터 행 수({len(rfm)})가 군집 수({n_clusters})보다 적습니다.")

    # ── 3. 표준화 + K-Means ───────────────────────────────────────
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(rfm_scaled)
    centers_scaled = kmeans.cluster_centers_

    # ── 4. 군집 레이블 자동 부여 ──────────────────────────────────
    cluster_types = _assign_cluster_type(centers_scaled)

    # ── 5. 결과 취합 ──────────────────────────────────────────────
    results = []
    for ci in range(n_clusters):
        mask = cluster_ids == ci
        subset = df[mask]
        ctype = cluster_types[ci]
        rule = CLUSTER_RULES[ctype]

        scatter_points = [
            {'name': row['_name'], 'r': round(row['R'], 1),
             'f': round(row['F'], 1), 'm': round(row['M'], 1)}
            for _, row in subset.iterrows()
        ]

        results.append({
            'cluster_index': ci,
            'label': ctype,
            'name': rule['name'],
            'description': rule['description'],
            'badge_color': rule['badge_color'],
            'count': int(mask.sum()),
            'avg_r': round(float(subset['R'].mean()), 1),
            'avg_f': round(float(subset['F'].mean()), 1),
            'avg_m': round(float(subset['M'].mean()), 1),
            'channel': rule['channel'],
            'keywords': rule['keywords'],
            'message': rule['message'],
            'scatter': scatter_points,
        })

    return {
        'total': len(df),
        'n_clusters': n_clusters,
        'clusters': results,
    }


def generate_sample_csv() -> str:
    """샘플 CSV 내용을 문자열로 반환합니다."""
    rows = [
        "고객명,마지막 방문일,구매 횟수,총 결제 금액",
        "김민준,2024-03-15,12,450000",
        "이서연,2024-03-20,8,320000",
        "박지호,2023-12-01,2,45000",
        "최수아,2024-03-25,15,680000",
        "정도윤,2024-01-10,3,75000",
        "강하은,2023-11-05,1,20000",
        "윤지우,2024-02-28,7,210000",
        "장서윤,2024-03-18,11,530000",
        "임현우,2023-10-20,2,38000",
        "한예진,2024-03-22,9,410000",
        "오민서,2024-01-05,4,95000",
        "신준혁,2024-03-10,13,590000",
        "류수현,2023-12-15,2,52000",
        "권지민,2024-03-28,16,720000",
        "남도현,2024-02-10,6,180000",
        "백채원,2023-09-01,1,15000",
        "유시온,2024-03-05,10,470000",
        "전나은,2024-01-20,5,130000",
        "송가온,2024-03-30,14,640000",
    ]
    return "\n".join(rows)
