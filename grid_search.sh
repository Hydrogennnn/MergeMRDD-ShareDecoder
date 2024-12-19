#!/bin/bash

# 设置固定的 epochs 数量
epochs=50

# 定义 mask-view-ratio 和 mask-ratio 的搜索范围
mask_view_ratios=(0.0 0.2 0.4 0.5 0.6 0.7 0.8 )
mask_ratios=(0.0 0.2 0.4 0.5 0.6 0.7 0.8)
config_path="./configs/emnist.yaml"

# 遍历所有的组合进行超参数搜索
for mv in "${mask_view_ratios[@]}"; do
    for m in "${mask_ratios[@]}"; do
        echo "Running with mask-view-ratio=${mv}, mask-ratio=${m}, epochs=${epochs}"

        # 执行你的训练脚本并传入参数
        python ./train_distent.py -f $config_path --mask-view-ratio $mv --mask-ratio $m --epochs $epochs
    done
done
