#training & test
CONFIG_PATH="downstream_clustering/clustering_config.py"

values=("train_118e_subseq" "119e24_subseq" "119e00_subseq" "119e_6_subseq")
for new_val in "${values[@]}"; do
    echo "Thay thế NAME_PTH_FILE thành \"$new_val\"..."
    sed -i 's/^NAME_PTH_FILE\s*=.*/NAME_PTH_FILE = "'"$new_val"'"/' "$CONFIG_PATH"
    grep "^NAME_PTH_FILE" clustering_config.py
    python run_exp_clustering.py -c rebar_ecg
    echo "---------------------------------------------"
done

