python lincls_bin.py \
	-a resnet50 \
	--train_list $1 \
	--val_list $2 \
	--world-size 1 \
	--num_classes 3 \
	--batch-size 64  \
	--pretrained $3 \
	--save-dir $4 \
	--fc_type 2 \
	--lr 0.003 \
	--cos_lr \
	--wd 0.0005 \
	--epochs 30 \
	--gpu $5 \