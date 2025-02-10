
# Install requirements
pip install -r requirements.txt

# Install apex
git clone https://github.com/NVIDIA/apex
cd apex; pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./

# Specify model type
MODEL_NAME=ViT-B_32

# Download pretrained model
wget https://storage.googleapis.com/vit_models/imagenet21k/${MODEL_NAME}.npz -P checkpoint
#wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/${MODEL_NAME}.npz -P checkpoint

# Train on new dataset
python3 src/train-test.py \
  --name 2012_B_32_1 \
  --dataset hymenoptera \
  --model_type ViT-B_32 \
  --pretrained_dir checkpoint/ViT-B_32.npz.1 \
  --fp16 \
  --fp16_opt_level O2 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --num_steps 50 \
  --warmup_steps 5 \
  --eval_every 10


# Test
python3 src/train-test.py \
  --name test1 \
  --model_type ${MODEL_NAME} \
  --pretrained_dir checkpoint/${MODEL_NAME}.npz \
  --fp16 \
  --fp16_opt_level O2 \
  --train_batch_size 128 \
  --num_steps 10 \
  --dataset hymenoptera \
  --test