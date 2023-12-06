nvidia-smi

# - - - - - - - -     Testing      - - - - - - - # 
expname="ACDC_runs"
version="3"
numlb=3 # 3, 7, 14
gpuid=9

python3 ./code/test_performance_2d.py \
    --root_path ../../data/ACDC \
    --res_path ./results/ACDC \
    --gpu_id=${gpuid} \
    --exp ${expname}/v${version} \
    --flag_check_with_best_stu \
    --labeled_num ${numlb}
