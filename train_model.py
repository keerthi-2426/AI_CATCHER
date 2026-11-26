from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 classification model
    # We use YOLOv8n-cls for classification tasks
    model = YOLO("yolov8n-cls.pt")  

    # Train the model
    # dataset needs to be the root folder containing 'train' and 'val' subdirectories
    print("Starting training...")
    results = model.train(
        data=r"e:\AI_Catcher\dataset",
        epochs=10, # Kept short for quick testing, user can easily bump this to 30 or 50
        imgsz=256, # reduced from 640 for faster training
        project="runs/classify",
        name="violence_classifier"
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
