from pathlib import Path
import json,hashlib
import numpy as np
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
BASE=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
CODE=Path('/home/chenkejun/CVPR/experiments/offline_mask_filters_196_20261023')
OUT=Path('/data/chenkejun/CVPR/results/offline_mask_filters_196_20261023')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES={'source':'来源资格筛选','boundary':'人工裁剪边缘','noise':'噪声筛选','grouping':'medoids分簇剔除重复','semantic':'背景语义剔除','covisible_iou_only':'共视 IoU 去重','full_support_containment':'包含关系分组（仅关联）','skip_repair_B':'90/85覆盖剔除','pr_filter':'p/r 与语义豁免','retained':'最终保留'}
