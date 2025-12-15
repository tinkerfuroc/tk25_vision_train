from dotenv import load_dotenv

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.training import train_detector

load_dotenv()


def run_finetuning() -> None:
    config = VisionConfig()
    best_model, results_dir = train_detector(config)
    print(f"Training finished. Results saved to {results_dir}")
    if best_model:
        print(f"Best model copied to {best_model}")


if __name__ == "__main__":
    run_finetuning()
