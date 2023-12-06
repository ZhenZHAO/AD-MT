# # - - - - - - - -      Testing      - - - - - - - # 

nvidia-smi

##############################################################

# # - - - - - - - - - - - - - - - - - - - - - # 
# #                   Pancrease
# # - - - - - - - - - - - - - - - - - - - - - # 

expname="Pancrease_runs"
version="1"
numlb=6 # 6, 12
gpuid=0

python3 ./code/test_performance_3d.py \
    --root_path ../../data/Pancreas/ \
    --res_path ./results/Pancreas/ \
    --dataset "Pancreas" \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --flag_check_with_best_stu
