import numpy as np
import numpy.typing as npt
import pytest

from personcv.models.Yunet import Yunet


class TestYunet:
    """Test class for Yunet face detection model."""

    @pytest.fixture
    def yunet_model(self) -> Yunet:
        """Fixture to provide a Yunet model instance."""
        return Yunet(conf_threshold=0.6, nms_threshold=0.3, top_k=5000, is_int8=False)

    def test_yunet_initialization(self) -> None:
        """Test Yunet model initialization."""
        model = Yunet()
        assert model.conf_threshold == 0.6
        assert model.nms_threshold == 0.3
        assert model.top_k == 5000
        assert model.is_int8 is False
        assert model.model is not None

    def test_yunet_initialization_with_custom_params(self) -> None:
        """Test Yunet model initialization with custom parameters."""
        model = Yunet(conf_threshold=0.8, nms_threshold=0.4, top_k=1000, is_int8=True)
        assert model.conf_threshold == 0.8
        assert model.nms_threshold == 0.4
        assert model.top_k == 1000
        assert model.is_int8 is True
        assert model.model is not None

    def test_preprocess(
        self, yunet_model: Yunet, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test the preprocessing method."""
        preprocessed = yunet_model.preprocess(sample_image)
        assert preprocessed.shape == sample_image.shape
        np.testing.assert_array_equal(preprocessed, sample_image)

    def test_inference(
        self, yunet_model: Yunet, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test the inference method."""
        preprocessed = yunet_model.preprocess(sample_image)
        output = yunet_model.inference(preprocessed)

        # Output should be a numpy array
        assert isinstance(output, np.ndarray)
        # Output should have float32 dtype
        assert output.dtype == np.float32
        # If faces are detected, output should have shape (n, 15)
        # where n is number of faces
        # 15 = 4 (bbox) + 10 (landmarks) + 1 (confidence)
        if output.size > 0:
            assert output.ndim == 2
            assert output.shape[1] == 15

    def test_postprocess(
        self, yunet_model: Yunet, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test the postprocess method."""
        preprocessed = yunet_model.preprocess(sample_image)
        inference_output = yunet_model.inference(preprocessed)
        result = yunet_model.postprocess(sample_image, inference_output)

        # Result should be a numpy array with the same shape as input
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype

    def test_full_pipeline(
        self, yunet_model: Yunet, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test the full inference pipeline."""
        # Step 1: Preprocess
        preprocessed = yunet_model.preprocess(sample_image)

        # Step 2: Inference
        inference_output = yunet_model.inference(preprocessed)

        # Step 3: Postprocess
        result = yunet_model.postprocess(sample_image, inference_output)

        # Verify the pipeline produces valid output
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype

    def test_decode_face_static_method(
        self, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test the static decode_face method."""
        # Create mock face detection output
        # Format: [x, y, w, h, landmark1_x, landmark1_y, ...,
        # landmark5_x, landmark5_y, confidence]
        mock_faces = np.array(
            [
                [
                    100,
                    100,
                    200,
                    200,
                    150,
                    120,
                    180,
                    120,
                    165,
                    140,
                    140,
                    160,
                    185,
                    160,
                    0.9,
                ]
            ],
            dtype=np.float32,
        )

        result = Yunet.decode_face(sample_image.copy(), mock_faces)

        # Result should be a numpy array with the same shape as input
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype

    def test_decode_face_empty_output(
        self, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test decode_face with empty output."""
        empty_faces = np.empty((0, 15), dtype=np.float32)
        result = Yunet.decode_face(sample_image.copy(), empty_faces)

        # Result should be the same as input when no faces are detected
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_repr(self) -> None:
        """Test the string representation of Yunet."""
        model = Yunet(conf_threshold=0.7, nms_threshold=0.4, top_k=1000, is_int8=True)
        repr_str = repr(model)

        assert "Yunet(" in repr_str
        assert "conf_threshold=0.7" in repr_str
        assert "nms_threshold=0.4" in repr_str
        assert "top_k=1000" in repr_str
        assert "is_int8=True" in repr_str

    def test_model_download_and_creation(self) -> None:
        """Download the model file and create the OpenCV model."""
        model = Yunet()

        # Check that the model attribute exists and is not None
        assert hasattr(model, "model")
        assert model.model is not None

        # Check that it's an OpenCV FaceDetectorYN instance
        assert hasattr(model.model, "detect")
        assert hasattr(model.model, "setInputSize")
