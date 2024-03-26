model='crysformer_v10'
dataset='megnet'
target='e_form'

batch_size=32
epoch=100
lr=4e-4
wd=1e-5

job="${model}-${dataset}-${target}-${epoch}e-lr${lr}-wd${wd}"

torchrun --nproc_per_node=1 main_crysformer.py \
    --batch_size ${batch_size} \
    --epochs ${epoch} \
    --inputs graph line_graph\
    --targets ${target} \
    --model ${model} \
    --lr ${lr} --weight_decay ${wd} \
    --warmup_epochs 5 \
    --dataset ${dataset} --data_path './data' \
    --n_train_val_test 60000 5000 4239 \
    --dist_eval \
    --output_dir "./output/${job}" --log_dir "./log/${job}" \
    # --pretrain "./output/crysformer_v10-megnet-e_form-100e-lr4e-4-wd1e-5/checkpoint-99.pth"