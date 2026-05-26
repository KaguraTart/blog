---
title: "UAV の経路計画のための競合解決アルゴリズムのレビュー"
description: "古典的なアルゴリズムから最先端の進行システムのレビューまで、幾何学的手法、最適化手法、複数マシンのコラボレーションと学習手法をカバーする、マルチ UAV システムにおける競合の特定および解決アルゴリズムの詳細な分析"
pubDate: 2026-04-07T11:12:59+08:00
tags: ["ドローン", "パスの計画", "競合の解決", "マルチエージェント", "アルゴリズムの概要"]
category: Tech
sourceHash: "bdea72e467b5ee1ca0825e4536706f6e89e09f1b"
---

# UAV 経路計画の競合解決アルゴリズムの概要

> 無人航空機 (UAV) が単一機の運用からクラスター連携に進化するにつれて、複数の UAV が同じ空域でタスクを実行する場合、経路の競合は避けられない中心的な問題となっています。 **紛争解決** とは、各ドローンの軌道や意思決定を調整して紛争状態を解消し、飛行の安全を確保しながらミッションを継続的に完了することを指します。この記事では、幾何学的な手法から深層強化学習に至るまで、競合の特定と解決のための主流のアルゴリズム フレームワークを体系的に整理し、各テクノロジーの核となる考え方、利点、欠点、進化を探ります。

---

## 1. 競合の定義と分類

### 1.1 パスの競合とは何ですか?

マルチ UAV システムでは、**競合** は、2 つ以上の UAV が時空次元で同じ空間位置 (または安全な隔離距離未満) を同時に占有している状態を指します。正式には:

$$
\exists \, i, j, \, i \neq j, \quad \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| < d_{安全}
$$

このうち、$\mathbf{p}_i(t)$ は $i$ 番目のドローンの位置、$d_{safe}$ は安全な隔離距離 (ミッション シナリオに応じて通常 5 ～ 50m) です。

### 1.2 紛争の分類

|タイプ |説明 |典型的なシナリオ |
|------|------|-----------|
| **宇宙紛争** |軌道は空間で交差します |ルートの交差、反対便 |
| **時間と空間の対立** |軌跡は時間次元で重なり合う |次々と同じ空域に進入 |
| **速度の競合** |相対速度が安全しきい値を超えています |追い上げシナリオ |
| **激しい紛争** |縦方向の対立 |交差点を持ち上げる |
| **動的競合** |移動障害物による衝突 (他の航空機) |空中遭遇 |

### 1.3 競合の指標

- **競合までの時間**: 競合が発生するまでの残り時間を予測します。
- **紛争確率**: 不確実性を考慮した紛争リスク評価
- **最小分離距離**: 軌道間の最も近い距離
- **解決時間**: 解決アクションが有効になるまでに必要な時間

---

## 2. 競合識別アルゴリズム競合の特定は競合を解決するための予備ステップであり、その中心となるのは**競合予測**であり、実際に競合が発生する前に競合が発生するかどうかを判断します。

### 2.1 幾何学的予測手法

最も直感的な方法は、幾何学的計算に基づく空間検出です。

```python
import numpy as np

def detect_conflict_2D(traj_i, traj_j, safe_radius=5.0):
    """
    检测两条轨迹是否发生空间冲突
    
    traj_i, traj_j: shape (N, 3) 的轨迹数组，每行为 (x, y, z)
    safe_radius: 安全隔离距离 (m)
    返回: (是否冲突, 最小间隔距离, 冲突时间点索引)
    """
    min_dist = float('inf')
    conflict_time = -1
    
    for t in range(len(traj_i)):
        dist = np.linalg.norm(traj_i[t] - traj_j[t])
        if dist < min_dist:
            min_dist = dist
            if dist < safe_radius:
                conflict_time = t
    
    is_conflict = min_dist < safe_radius
    return is_conflict, min_dist, conflict_time
```

### 2.2 速度障害

**速度障害物 (VO)** は、ロボット工学分野で最も古典的な衝突検出および予測方法です。これは、Fioretti & Fraichard (1999) によって UAV 分野に導入されました。

核となるアイデア: スピード空間に「禁止エリア」を構築する。ドローンの現在の速度ベクトルがこの領域内にある場合、間違いなく衝突が発生します。

$$
VO_{ij} = \{ \mathbf{v} \mid \lambda(\mathbf{p}_j - \mathbf{p}_i, \mathbf{v} - \mathbf{v}_j) \cap D_{ij} \neq \varnothing \}
$$

ここで、$D_{ij}$ は $\mathbf{p}_j - \mathbf{p}_i$ を軸、半径を安全距離とする円柱、$\lambda$ は半光線です。

```python
def velocity_obstacle(p_i, v_i, p_j, v_j, r_safe=5.0):
    """
    计算第 i 架 UAV 的速度障碍区域
    p_i, p_j: 位置向量
    v_i, v_j: 速度向量
    r_safe: 安全半径
    """
    rel_pos = p_j - p_i
    rel_vel = v_i - v_j
    dist = np.linalg.norm(rel_pos)
    
    if dist == 0:
        return None
    
    # 相对位置的夹角（安全圆柱的视角）
    theta = np.arcsin(r_safe / dist)
    
    # 障碍扇区的两条边向量
    dir_pos = rel_pos / dist
    perp_dir = np.array([-dir_pos[1], dir_pos[0]])
    
    # 两条边界速度向量
    v_left  = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos + np.sin(theta) * perp_dir)
    v_right = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos - np.sin(theta) * perp_dir)
    
    return v_left, v_right  # VO 的两条边界

def is_in_vo(v_i, v_left, v_right, v_j):
    """判断速度 v_i 是否落在 VO 区域内"""
    # 转换到相对坐标系
    rel_v = v_i - v_j
    rel_left  = v_left - v_j
    rel_right = v_right - v_j
    
    # 检查 rel_v 是否在 rel_left 和 rel_right 之间
    cross_left  = np.cross(rel_left,  rel_v)
    cross_right = np.cross(rel_right, rel_v)
    
    return np.sign(cross_left) == np.sign(cross_right)
```

### 2.3 不確実性を認識した競合検出

実際のシステムでは、位置情報には GPS 誤差やセンサーノイズなどの不確実性が含まれることがよくあります。 **確率的競合検出** 競合判断に確率分布を導入します。

$$
P_{競合} = \int\int \mathbb{1}(\| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe}) \cdot f_i(\mathbf{p}_i) \cdot f_j(\mathbf{p}_j) \, d\mathbf{p}_i \, d\mathbf{p}_j
$$

ここで、$f_i、f_j$ は、位置の確率密度関数 (通常はガウス分布であると想定されます) です。 $P_{conflict} > P_{threshold}$ の場合、競合アラームがトリガーされます。一般的な方法には次のようなものがあります。
- **モンテカルロ サンプリング**: 多数の確率分布をサンプリングした後の競合率の統計
- **線形検証ツール (LVT)**: ガウス分布の仮定に基づく確率矛盾の分析的近似
- **確率的到達可能セット**: 確率的制御理論に基づいたセット表現

---

## 3. 競合解決アルゴリズム

### 3.1 幾何学的手法

#### 3.1.1 Rate Obstacle 法 (Rate Obstacle / VO 補正)

VO ベースの排除戦略: VO エリアを回避できる目標速度 $\mathbf{v}_{new}$ を見つけます。

```python
def vo_resolution(p_i, v_i, p_j, v_j, v_max=20.0, r_safe=5.0):
    """
    基于 Velocity Obstacle 的冲突消解
    返回满足速度约束且避开 VO 的新速度
    """
    vo = velocity_obstacle(p_i, v_i, p_j, v_j, r_safe)
    if vo is None:
        return v_i
    
    v_left, v_right = vo
    
    # 所有候选速度（速度空间中均匀采样）
    best_v = v_i
    min_dist_to_vo = float('inf')
    
    for speed in np.linspace(0, v_max, 20):
        for angle in np.linspace(0, 2*np.pi, 36):
            v_candidate = speed * np.array([np.cos(angle), np.sin(angle)])
            
            # 跳过落在 VO 内的速度
            if is_in_vo(v_candidate, v_left, v_right, v_j):
                continue
            
            # 选择最接近原始速度方向且最"远离"VO 的速度
            dist_to_original = np.linalg.norm(v_candidate - v_i)
            # 到 VO 边界的距离
            dist_to_vo = min(
                np.linalg.norm(v_candidate - v_left),
                np.linalg.norm(v_candidate - v_right)
            )
            
            # 优化目标：尽量接近原速度，同时远离 VO
            score = dist_to_vo - 0.5 * dist_to_original
            if dist_to_vo > min_dist_to_vo and dist_to_vo > 1.0:
                min_dist_to_vo = dist_to_vo
                best_v = v_candidate
    
    return best_v
```

#### 3.1.2 人工ポテンシャル場法（人工ポテンシャル場）

ドローンを「ポテンシャル場」内を移動する荷電粒子と考えてください。
- **ターゲットポイント**が魅力を生み出す
- **障害物/他のドローン**は反発力を生成します

$$
\mathbf{F}_{合計} = \mathbf{F}_{att} + \sum_j \mathbf{F}_{rep,j}
$$

その中には:
$$
\mathbf{F}_{att} = k_{att} \cdot (\mathbf{p}_{goal} - \mathbf{p}_i)
$$
$$
\mathbf{F}_{rep,j} = k_{rep} \cdot \frac{\mathbf{p}_i - \mathbf{p}_j}{\| \mathbf{p}_i - \mathbf{p}_j \|^3} \cdot (\| \mathbf{p}_i - \mathbf{p}_j \| - d_{safe})
$$

**利点**: 計算が速く、リアルタイム制御に適しています。
**短所**: 極小値に陥りやすい (2 つのドローンが互いに「押すことができない」場合、振動します)

**改善の方向性**:
- ポテンシャル フィールドの整形: 極小値を避けるためにポテンシャル フィールドの形状を調整します。
- 複数の仮想ポテンシャルフィールド: 仮想障害物を導入してトラップエリアの周囲に経路を案内します
- ハイブリッド手法: A* または RRT* と組み合わせ、ポテンシャル場を使用して局所微調整を行います。

#### 3.1.3 ボロノイ図法ボロノイ図は空間を複数のエリアに分割するために使用され、各ドローンはそのボロノイ ユニット内を飛行します。これにより、他のドローンまでの距離が常にボロノイ境界までの距離よりも大きくなります。

1. 現時点のボロノイ図をリアルタイムで構築する
2. 各ドローンは、ボロノイ ユニットの最遠点 (最適点) に近いウェイポイントを選択します。
3. ウェイポイントの方向に移動し、競合が検出された場合は新しいボロノイ パスに切り替えます。

```python
from scipy.spatial import Voronoi
import numpy as np

def voronoi_resolution(positions, v_i, p_i, v_max=20.0):
    """
    基于 Voronoi 图的多机冲突消解
    positions: 所有无人机位置 (N, 2)
    """
    vor = Voronoi(positions)
    
    # 找到当前无人机 i 的 Voronoi 单元
    region_idx = vor.point_region[np.where(vor.point_region == vor.point_region[0])[0][0]]
    # 简化的 Voronoi 路径选择：取 Voronoi 顶点的方向
    vertices = [vor.vertices[v] for v in vor.regions[region_idx] if v >= 0]
    
    if not vertices:
        return v_i
    
    # 选择最接近原始速度方向的安全顶点
    best_dir = v_i / np.linalg.norm(v_i)
    best_vertex = None
    max_projection = -float('inf')
    
    for v in vertices:
        direction = v - p_i
        if np.linalg.norm(direction) < 0.01:
            continue
        direction = direction / np.linalg.norm(direction)
        projection = np.dot(direction, best_dir)
        
        if projection > max_projection:
            max_projection = projection
            best_vertex = v
    
    if best_vertex is None:
        return v_i
    
    return np.clip(best_vertex - p_i, -v_max, v_max)
```

### 3.2 最適化手法

#### 3.2.1 混合整数線形計画法 (MILP)

MILP は、複数の UAV の軌道計画を数学的最適化問題として形式化する古典的なフレームワークであり、Shouwenaars らによって UAV 分野で先駆的に開発されました。 (2001)。

**核となるアイデア**: 連続軌道は区分多項式または固定ウェイポイント シーケンスで表され、競合制約と安全制約は線形不等式で表され、飛行セグメントの切り替えロジックを表すために 2 進整数変数が導入されます。

$$
\min \quad \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k-1} \|^2 + \lambda \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k}^{pref} \|^2
$$

**制約**:
- 運動学的制約: $\| \mathbf{v}_{i,k} \| \leq v_{max}$, $\| \mathbf{a}_{i,k} \| \leq a_{max}$
- 競合回避の制約:
  - $\| の場合\mathbf{p}_{i,k} - \mathbf{p}_{j,k} \| < d_{safe}$ の場合、対応するバイナリ変数 $\delta_{ijk} = 1$
  - OR 制約を導入します: $\sum_j \delta_{ijk} \leq 0$ (すべての $\delta$ を強制的に 0、つまり競合しないようにします)
- タスク完了の制約: $\| \mathbf{p}_{i,K} - \mathbf{p}_{goal,i} \| < \バレプシロン$

```python
# 概念性 MILP 冲突约束（伪代码）
"""
minimize: sum_i sum_k (v_i,k - v_pref_i,k)^2

subject to:
    for each UAV i, segment k:
        p_i,k+1 = p_i,k + v_i,k * dt          # 运动学
        norm(v_i,k) <= v_max                   # 速度限幅
        norm(a_i,k) <= a_max                   # 加速度限幅
        
    for each UAV pair (i,j), segment k:
        norm(p_i,k - p_j,k)^2 >= d_safe^2 OR delta_ik = 0
        M * delta_ik >= norm(p_i,k - p_j,k)^2 - d_safe^2
        sum_j delta_jk <= 0                     # 所有 delta 必须为 0
"""
```**利点**: 厳しい制約が確実に満たされる、グローバル最適ソリューション
**短所**: MILP ソルバー (CPLEX、Gurobi) の計算の複雑さはドローンの数に応じて指数関数的に増加するため、5 ～ 10 台を超えるドローンを使用したシナリオをリアルタイムで解決することが困難になります。

#### 3.2.2 動的ウィンドウアプローチ (DWA)

ロボットの動作計画から借用された DWA は、速度空間 $(v, \omega)$ をサンプリングし、各候補速度を評価します。
1. **ターゲットに向かう軌道**
2. **衝突安全性** (短期軌道のシミュレーションにより判断)
3. **速度到達性**

```python
def dwa_resolution(p_i, v_i, v_goal, obstacles,
                   v_max=3.0, v_min=0.0,
                   a_max=2.0, dt=0.1, predict_time=2.0,
                   safe_radius=1.5):
    """
    Dynamic Window Approach 用于 UAV 冲突消解
    """
    # 1. 构建动态窗口（当前可达速度集）
   Vw = []
    for v in np.arange(max(0, v_i[0] - a_max*dt), min(v_max, v_i[0] + a_max*dt), 0.1):
        for w in np.arange(v_i[1] - a_max*dt, v_i[1] + a_max*dt, 0.1):
            Vw.append((v, w))
    
    best_score = -float('inf')
    best_v = v_i
    
    for (v, w) in Vw:
        # 2. 预测轨迹
        traj = []
        p_pred = p_i.copy()
        v_pred = np.array([v, w])
        for t in np.arange(0, predict_time, dt):
            traj.append(p_pred.copy())
            p_pred = p_pred + v_pred * dt
        
        # 3. 碰撞检测
        collision = False
        for p_obs in obstacles:
            for p_t in traj:
                if np.linalg.norm(p_t - p_obs) < safe_radius:
                    collision = True
                    break
            if collision:
                break
        
        if collision:
            continue
        
        # 4. 评分函数
        score_heading = np.linalg.norm(p_pred - v_goal)  # 越小越好
        score_velocity = v  # 越大越好（偏好高速）
        score_clearance = min([np.linalg.norm(p_t - p_obs)
                               for p_obs in obstacles for p_t in traj])
        
        total_score = (
            2.0 * (1.0 / (score_heading + 1e-6)) +
            1.0 * score_velocity +
            0.5 * score_clearance
        )
        
        if total_score > best_score:
            best_score = total_score
            best_v = np.array([v, w])
    
    return best_v
```

#### 3.2.3 分散モデル予測制御 (DMPC)

DMPC は、大規模なマルチ UAV 群の主流の方法であり、各 UAV は次のことを行います。
1. ローカル情報と近隣コミュニケーションに基づいてローカル予測モデルを構築する
2. 有限時間領域での局所最適化問題を解く
3. 制御の最初のステップのみを実行し、その後ローリング再最適化を実行します。

**主要な一貫性制約**:
$$
\sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_i[k+k_p|k] - \mathbf{x}_j[k+k_p|k]) = 0, \quad \forall k_p \in \{1, \dots, N_p\}
$$

ここで、$\mathcal{N}_i$ は UAV $i$ の近傍のセット、$a_{ij}$ は隣接行列の重みです。

DMPC の主な利点は **スケーラビリティ** です。各ドローンは近隣のドローンと通信するだけで済み、グローバルなドローンの数に応じて計算量が急激に増加することはありません。

### 3.3 複数マシンの連携方法

#### 3.3.1 グラフ理論に基づく一貫性アルゴリズム

マルチ UAV システムをグラフ $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ としてモデル化します。
- ノード $v_i \in \mathcal{V}$ はドローンを表します
- エッジ $e_{ij} \in \mathcal{E}$ は通信リンクを表します

**コンセンサスプロトコル**:
$$
\dot{\mathbf{x}}_i = \sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_j - \mathbf{x}_i)
$$競合解決に適用される場合、各ドローンの **優先度** または **コスト関数** が状態変数として使用され、コンセンサス収束後に解決アクションが選択されます。

一般的なトポロジの比較:
- **最近傍トポロジ**: $\mathcal{O}(N)$ トラフィックが発生しますが、収束が遅い
- **完全接続トポロジ**: 高速コンバージェンスですが、$\mathcal{O}(N^2)$ トラフィックが発生します
- **メトロポリス-ヘイスティングスの重み付け**: コンバージェンス速度と通信オーバーヘッドのバランスをとります

#### 3.3.2 市場ベース

バイオニック アルゴリズムのアイデア: 紛争地域を「資源」として扱い、各ドローンはオークションを通じてその資源を使用する権利を競います。
1. **入札**: 各ドローンは、自身のミッションの緊急性とコストを計算します。
2. **オークション**: 最高入札者が優先権を獲得し、他のドローンは待機するか迂回します
3. **決済 (割り当て)**: リソース割り当てテーブルを更新し、競合がなくなるまで繰り返します。

```python
import heapq

def auction_based_resolution(uavs, conflict_zone, max_iterations=20):
    """
    基于拍卖的多机冲突消解
    uavs: List[UAV] - 无人机列表
    conflict_zone: 冲突区域中心及半径
    """
    allocation = {}  # zone_id -> winner_uav_id
    unallocated = list(uavs)
    
    for iteration in range(max_iterations):
        if not unallocated:
            break
        
        # 每次拍卖冲突区域使用权
        bids = []
        for uav in unallocated:
            urgency = uav.task_deadline - time.now()
            cost = uav.compute_detour_cost(conflict_zone)
            bid = urgency / (cost + 1e-6)
            bids.append((bid, uav))
        
        # 最高出价者胜出
        bids.sort(reverse=True)
        winner_bid, winner = bids[0]
        
        allocation[conflict_zone.id] = winner.id
        unallocated.remove(winner)
        
        # 对非胜出者计算绕行路径
        for uav in unallocated:
            uav.compute_alternative_path(conflict_zone)
    
    return allocation
```

#### 3.3.3 ゲーム理論の手法

**非協力ゲーム**として競合解決をモデル化します:
- 各ドローンは**プレイヤー**です
- 各ドローンの軌道は**戦略**です
- 自身の紛争リスクと飛行コストを最小限に抑えることが **利益関数 (効用)**

**ナッシュ均衡** は、プレイヤーが一方的に戦略を変更することでより良い利益を得ることができない一連の戦略の組み合わせです。

$$
\forall i, \quad \mathbf{s}_i^* \in \arg\min_{\mathbf{s}_i \in \mathcal{S}_i}
\mathcal{J}_i(\mathbf{s}_i^*, \mathbf{s}_{-i}^*)
$$

**相関平衡**は、ナッシュ平衡よりも分散的に解くのが簡単で、UAV クラスターではより実用的です。

### 3.4 学習方法

#### 3.4.1 強化学習 (RL)

近年、**深層強化学習 (DRL)** により、UAV クラスターの競合解決が大幅に進歩しました。典型的なフレームワーク:- **状態空間 $\mathcal{S}$**: すべての UAV の位置、速度、目標点、障害物
- **アクションスペース $\mathcal{A}$**: 速度変更 $(\Delta v_x, \Delta v_y, \Delta v_z)$ または進行角変更
- **報酬関数 $\mathcal{R}$**:
  - 衝突: $r_{衝突} = -100$
  - 目標に近い: $r_{progress} = +10 \cdot \Delta dist$
  - 安全な距離を維持します: $r_{safety} = +5$ ($\|p_i - p_j\| > d_{safe}$ の場合)
  - エネルギー消費量: $r_{energy} = -0.1 \cdot \|\Delta v\|^2$

**MADDPG (マルチエージェント DDPG)** は、最も一般的に使用されるマルチマシン RL フレームワークの 1 つです。

```python
# MADDPG 核心思路（伪代码）
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim * n_agents, action_dim * n_agents)
        self.target_actor = copy_network(self.actor)
        self.target_critic = copy_network(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1e6)
    
    def select_action(self, state, noise=0.1):
        action = self.actor.forward(state)
        action += noise * np.random.randn(action.shape)
        return np.clip(action, -1, 1)
    
    def update(self, batch):
        # 从全局视角更新 Critic（这是 MADDPG 的关键创新）
        states, actions, rewards, next_states = batch
        
        # 目标网络更新
        target_actions = [self.target_actor.forward(ns) for ns in next_states]
        target_Q = self.target_critic.forward(
            torch.cat(states), torch.cat(target_actions))
        
        # 均值聚集（Mean Aggregation）：所有智能体的目标 Q 值取平均
        target_Q = sum(target_Q) / n_agents
        
        # 策略更新
        ...
```

#### 3.4.2 アテンションメカニズム (アテンション)

**グラフ アテンション ネットワーク (GAT)** は、UAV 間の相対的な重要性の関係をモデル化するために使用されます。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
    
    def forward(self, h, adj):
        """
        h: (N, D_in) - N 个节点的特征
        adj: (N, N) - 邻接矩阵
        """
        Wh = self.W(h)  # (N, D_out)
        N = Wh.size(0)
        
        # 计算注意力系数
        a_input = torch.cat([
            Wh.repeat(1, N).view(N, N, -1),
            Wh.repeat(N, 1).view(N, N, -1)
        ], dim=2)  # (N, N, 2*D_out)
        
        e = self.leaky_relu(self.a(a_input).squeeze(2))  # (N, N)
        
        # Mask 掉非邻接节点
        e = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(e, dim=1)  # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 特征聚集
        h_out = torch.matmul(attention, Wh)  # (N, D_out)
        return F.elu(h_out)
```

GAT を通じて、ドローンはどの近隣航空機が自らの決定に最も大きな影響を与えるかを適応的に学習することができ、それによって **ソフト コーディネーション**を実現します。すべてのドローンと通信する必要はなく、注意の重みが高い近隣航空機に注意を払うだけで済みます。

#### 3.4.3 模倣学習

エキスパートの軌跡 (DMPC または幾何学的手法からのソリューション) を使用してポリシー ネットワークをトレーニングします。

$$
\mathcal{L} = -\mathbb{E}_{(s,a) \sim d_{\pi^*}}[\log \pi_\theta(a \mid s)]
$$

**DAgger (データセット集約)** メソッドは、専門家の注釈付きデータを繰り返し収集して、分布シフトの問題を解決できます。

---

## 4. アルゴリズムの比較と選択ガイド|アルゴリズムカテゴリ |リアルタイム |スケーラビリティ |最適性 |不確実性の取り扱い |典型的なアプリケーション シナリオ |
|----------|----------|----------|----------|-----|---------------------|
| **VO / ジオメトリ** | ⭐⭐⭐⭐⭐ | ⭐⭐ |ローカル最適 | ❌ | 2 ～ 5 フレーム、キャッチアップの競合 |
| **ポテンシャル場法** | ⭐⭐⭐⭐⭐ | ⭐⭐ |局所最適 | ❌ |リアルタイムの障害物回避、動的障害物 |
| **ボロノイ** | ⭐⭐⭐ | ⭐⭐⭐ |ローカル最適 | ❌ |疎クラスタのパス計画 |
| **MILP** | ⭐ | ⭐⭐ | **世界最適** | ⚠️ スケーラブル | ≤10 ラックのオフライン計画 |
| **DMPC** | ⭐⭐⭐ | ⭐⭐⭐⭐ |準最適 | ⚠️ 内蔵 | 10 ～ 50 ラック クラスタ |
| **グラフ理論/オークション** | ⭐⭐⭐ | ⭐⭐⭐⭐ |準最適 | ❌ |大規模クラスターのタスク割り当て |
| **ゲーム理論** | ⭐⭐ | ⭐⭐⭐ |関連する均衡 | ⚠️ スケーラブル |競合シナリオ |
| **DRL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |最適な戦略 | ✅ 内蔵 | 50 以上のクラスター、エンドツーエンド |
| **GAT+RL** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |最適な戦略 | ✅ 内蔵 |超大規模異種クラスタ |

**選択の提案**:
- **5 以下**: VO または DMPC を優先します。計算は高速で、解の品質は保証されます。
- **5–50**: DMPC + グラフ コヒーレンス プロトコル、または MADDPG
- **50 ラック以上**: GAT + アテンション + RL、エンドツーエンド戦略が最も現実的な選択です
- **競合当事者**: ゲーム理論の枠組み (ナッシュ均衡/相関均衡)
- **重大な不確実性** (GPS エラー、風の乱れ): 確率的競合検出 + 堅牢な MPC

---

## 5. フロンティアの進歩と傾向

### 5.1 エンドツーエンドの強化学習2023 年に Google DeepMind によって公開された [SMARTS](https://github.com/hijkzzz/SMARTS) プラットフォームと [AlphaPilot](https://www.microsoft.com/en-us/research/project/alpha-pilot/) プロジェクトは、UAV クラスターでのエンドツーエンド RL の適用を促進し、認識から意思決定までの単一のポリシー ネットワークが生のセンサー データを直接処理できることを実証しました。

### 5.2 フェデレーテッド ラーニング

各ドローンのデータのプライバシーを保護することを前提として、複数のドローンのエクスペリエンスがフェデレーテッド ラーニングを通じて集約されます。
1. 各 UAV の現地訓練戦略
2. 生データではなくグラデーションのみを中央サーバーにアップロードします
3. 更新を集約した後、新しい戦略を発行する

UAV クラスターにおけるデータの分散とラベル取得の難しさの問題を解決します。

### 5.3 不確実な堅牢な MPC

近年、**チューブベースの MPC** と **シナリオベースの MPC** は、不確実性を有界摂動または確率的シナリオとしてモデル化し、最適化問題のロバスト性を明示的に制約します。

$$
\forall \omega \in \mathbb{W}: \quad \mathbf{x}[k+1] = A\mathbf{x}[k] + B\mathbf{u}[k] + E\omega[k]
$$

「不変集合」を事前計算して、最悪の場合でも安全制約が満たされるようにします。

### 5.4 多目的の競合解決

実際のタスクでは、競合解決では次のことも考慮する必要があります。
- **タスク完了率**: 単に作業を進めてタスクをタイムアウトさせないでください。
- **エネルギー消費**: 電力が限られている UAV は、追加の飛行距離を最小限に抑える必要があります。
- **通信遅延**: 分散システムにおける情報の遅れは誤った判断を引き起こす可能性があります
- **公平性**: 特定の UAV は常に譲歩できるわけではありません (飢えた問題)

**パレート最適フロンティア** 検索は、複数の目的の競合解決を解決するための中心的なツールです。

---

## 6. まとめ

UAV の経路計画における競合解決は、幾何学的計算、最適化理論、分散システム、機械学習にまたがる横断的な問題です。初期の幾何学的レート バリア手法から MILP グローバル最適化、分散 MPC および深層強化学習に至るまで、アルゴリズムの進化の中核となる原動力は常に次のとおりです。

> **より短い時間でより大きな不確実性の下で、より多くのドローンのより安全な軌道を見つける方法。 **将来のトレンドは **ハイブリッド アーキテクチャ** になります。つまり、学習手法を使用してローカルで迅速な意思決定を行い、最適化手法を使用してグローバルな軌道を検証し、通信プロトコルを使用して複数マシンのコラボレーションの一貫性を確保します。この 3 つを組み合わせることで、UAV 群の安全、効率的、スケーラブルな自律飛行を真に実現できます。

---

**参考文献** (時間順に並べ替え):1. van den Berg, J.、Lin, M.、および Manocha, D. (2008)。 *リアルタイム マルチエージェント ナビゲーションのための相互速度障害物。* IEEE ロボティクスとオートメーションに関する国際会議 (ICRA)。
2. リチャーズ、A.、ハウ、JP (2002)。 *混合整数線形計画法を使用した衝突回避を伴う航空機の軌道計画。* AIAA 誘導、航法、および制御会議 (GNC)。
3. アロンソ・モーラ、J.、他。 （2018年）。 *複数車両システム向けの最適化ベースの衝突回避。* IEEE Transactions on Robotics (TRO)。
4. Everett、M.、他。 （2021年）。 *深層強化学習による密集した交通における衝突回避* IEEE ロボティクスとオートメーションに関する国際会議 (ICRA)。
5. Zhou、M.、他。 （2019年）。 *乱雑な環境における UAV の経路計画に関する調査。* 高度道路交通システムに関する IEEE トランザクション (T-ITS)。
6. Lowe、R.、他。 （2017年）。 *協力と競争が混在する環境におけるマルチエージェントのアクター兼批評家 (MADD)PG).* 神経情報処理システム (NeurIPS) に関する会議。
7. Foerster、J.、他。 （2018年）。 *反事実的なマルチエージェント ポリシー勾配 (COMA)。* AAAI 人工知能会議。
8. ラシッド、T.、他。 （2018年）。 *QMIX: 深いマルチエージェント強化学習のための単調値関数因数分解。* 機械学習に関する国際会議 (ICML)。
9. Veličković、P.、他。 （2018年）。 *グラフ アテンション ネットワーク。* 学習表現に関する国際会議 (ICLR)。
10. ヤン、C.、他。 (2025年)。 *スケーラブルな固定翼 UAV 艦隊の衝突回避を伴う群集のための時空間的注意を伴うマルチエージェント強化学習。* 高度道路交通システム (T-ITS) に関する IEEE トランザクション。
11. Huo、D.、他。 （2023年）。 *障害物環境における UAV の無衝突モデル予測軌道追跡制御。* 航空宇宙および電子システムに関する IEEE トランザクション (TAES)。
12. ファン、T.、他。 （2020年）。 *ディ複雑なシナリオでのナビゲーションのための深層強化学習による分散型マルチロボット衝突回避* 国際ロボット研究ジャーナル (IJRR)。
13. Jiang、C.、他。 （2024年）。 *マルチロボットフォーメーションナビゲーションのための信念伝播による分散サンプリングベースのモデル予測制御。* IEEE Robotics and Automation Letters (RA-L)。
14. Goeckner、A.、他。 （2024年）。 *マルチロボット システムの回復力のある分散調整のためのグラフ ニューラル ネットワーク ベースのマルチエージェント強化学習。* インテリジェント ロボットおよびシステムに関する IEEE/RSJ 国際会議 (IROS)。