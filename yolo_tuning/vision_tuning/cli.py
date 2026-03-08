from typing import Optional

from yolo_tuning.vision_tuning.commands import build_parser, run_cli


def main(argv: Optional[list] = None) -> None:
    run_cli(argv)


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
