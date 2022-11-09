DATA_ROOT_PATH=datasets
TRAINER=ADG
DATASET=ssdg_pacs  # art_painting, cartoon, photo, sketch
NLAB=-5 # annotation budget 5%, -100~0 represents ratio
SEED=5

CUDA_VISIBLE_DEVICES=0 python train_ceg.py \
--root ${DATA_ROOT_PATH} \
--seed ${SEED} \
--trainer ${TRAINER} \
--update-data 1 \
--output-dir output-pacs-art_painting-seed-${SEED} \
--source-domains sketch photo cartoon \
--target-domains art_painting \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${DATASET}_best.yaml \
DATASET.NUM_LABELED ${NLAB}
# --resume \