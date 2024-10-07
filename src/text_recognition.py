import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image

# Load pre-trained models
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Custom dataset class
class HandwrittenEquationDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor, tokenizer, max_length=128):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze()

        # Tokenize label
        labels = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length).input_ids.squeeze()

        return pixel_values, labels

# Prepare dataset
train_image_paths = ["path/to/train/image1.png", "path/to/train/image2.png", ...]
train_labels = ["equation1", "equation2", ...]
val_image_paths = ["path/to/val/image1.png", "path/to/val/image2.png", ...]
val_labels = ["equation1", "equation2", ...]

train_dataset = HandwrittenEquationDataset(train_image_paths, train_labels, feature_extractor, tokenizer)
val_dataset = HandwrittenEquationDataset(val_image_paths, val_labels, feature_extractor, tokenizer)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")