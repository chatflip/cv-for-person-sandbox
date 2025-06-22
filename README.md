# pcv

## Requirement

- [uv](https://docs.astral.sh/uv/)

## Usage

### For Users

```bash
# Run inference with default model
uv run python demo/inference_image.py
uv run python demo/inference_video.py
uv run python demo/inference_webcam.py

# Run inference with specific model
uv run python demo/inference_image.py --arch KeypointRCNN
uv run python demo/inference_video.py --arch KeypointRCNN
uv run python demo/inference_webcam.py --arch KeypointRCNN
```

### For Developers

#### Install development dependencies

```bash
uv sync --group dev
```

#### Code Quality

```bash
# Run linter and formatter
uv run ruff check
uv run ruff format

# Run type checker
# uv run mypy

# Format markdown files
uv run mdformat README.md
```

#### Testing

```bash
# Run all tests
uv run pytest
```

## License

- assets/sample_image.jpg from [Unsplash](https://unsplash.com/ja/%E5%86%99%E7%9C%9F/%E5%A5%B3%E6%80%A7%E3%81%A8%E7%99%BD%E9%A6%AC%E3%81%8C%E4%B8%80%E7%B7%92%E3%81%AB%E3%83%9D%E3%83%BC%E3%82%BA%E3%82%92%E3%81%A8%E3%81%A3%E3%81%A6%E3%81%84%E3%81%BE%E3%81%99-EFxfRtR5lXs)
- assets/sample_video.mp4 from [Pexels](https://www.pexels.com/video/woman-doing-a-jump-rope-exercise-2785536/)

## Author

[chatflip](https://github.com/chatflip)
