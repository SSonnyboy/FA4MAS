#!/bin/bash

# ===================== 配置 =====================
MODEL="Qwen/Qwen2.5-7B-Instruct"
API_KEY="none"
BASE_URL="http://localhost:9001/v1"

# 数据集
DATASETS=(
    "data/repo/Who&When/Algorithm-Generated"
    "data/repo/Who&When/Hand-Crafted"
)

# 策略 (对应代码中的 --strategy 参数)
STRATEGIES=(
    "reverse_trace"
    "window_summary"
    "graph"
)

# 开始全量运行
for ds in "${DATASETS[@]}"; do
    ds_name=$(basename "$ds")

    for strategy in "${STRATEGIES[@]}"; do

        # ================== 1. 开启 GT ==================
        out_gt="outputs/${ds_name}_${strategy}_GT_ON"
        echo "🔥 运行：$ds | 策略: $strategy | GT=ON"
        python tot_failure_attribution2.py \
            --strategy "$strategy" \
            --model "$MODEL" \
            --data_dir "$ds" \
            --api_key "$API_KEY" \
            --base_url "$BASE_URL" \
            --use_ground_truth \
            --output_dir "$out_gt"

        # ================== 2. 关闭 GT ==================
        out_no_gt="outputs/${ds_name}_${strategy}_GT_OFF"
        echo "🔥 运行：$ds | 策略: $strategy | GT=OFF"
        python tot_failure_attribution2.py \
            --strategy "$strategy" \
            --model "$MODEL" \
            --data_dir "$ds" \
            --api_key "$API_KEY" \
            --base_url "$BASE_URL" \
            --output_dir "$out_no_gt"

    done
done

echo "🎉 所有任务完成！所有结果已保存在 outputs/ 目录"
echo "✅ GT ON  和  GT OFF 结果完全分开，不会覆盖！"
echo "✅ 每个任务都有独立文件 + SUMMARY"