eval "$(conda shell.bash hook)"

# git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
conda activate diffusers

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --instance_data_dir="./data/dog" \
  --class_data_dir="class_dir" \
  --output_dir="class-based-output" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="adamsmith" \
  --class_prompt="person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --use_8bit_adam \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800

# python inference.py "fine-tuned-model-output/800" "a photo of Nader wearing sunglasses"