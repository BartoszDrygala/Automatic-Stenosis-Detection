import cv2
import matplotlib.figure
import numpy as np
import pytest

import main


class _FakeFigure:
    """Stand-in for the objects main() calls .show() on."""
    def __init__(self):
        self.shown = False

    def show(self):
        self.shown = True


class TestFindStenosis:
    def test_mask_only_returns_three_visualisations(self, stenotic_mask):
        stenosis_img, overlay, paths_img = main.find_stenosis(stenotic_mask)
        assert isinstance(stenosis_img, matplotlib.figure.Figure)
        assert isinstance(overlay, matplotlib.figure.Figure)
        assert hasattr(paths_img, "gcf")  # the pyplot module

    def test_with_image_produces_the_side_by_side_versions(self, stenotic_mask):
        image = np.random.default_rng(0).integers(0, 255, stenotic_mask.shape, dtype=np.uint8)
        stenosis_img, overlay, paths_img = main.find_stenosis(stenotic_mask, image)
        assert len(stenosis_img.axes) == 2
        assert len(overlay.axes) == 2
        assert len(paths_img.gcf().axes) == 4

    def test_healthy_vessel_still_produces_output(self, healthy_mask):
        stenosis_img, overlay, _ = main.find_stenosis(healthy_mask)
        assert len(stenosis_img.axes) == 1
        assert len(overlay.axes) == 1

    def test_stenosis_is_drawn_for_a_stenotic_vessel_only(self, healthy_mask, stenotic_mask):
        def red_pixels(mask):
            fig = main.find_stenosis(mask)[0]
            shown = np.asarray(fig.axes[0].images[0].get_array())
            return int(((shown[..., 0] == 255) & (shown[..., 1] == 0)).sum())

        assert red_pixels(healthy_mask) == 0
        assert red_pixels(stenotic_mask) > 0

    def test_input_mask_is_not_modified(self, stenotic_mask):
        before = stenotic_mask.copy()
        main.find_stenosis(stenotic_mask)
        assert np.array_equal(stenotic_mask, before)

    def test_accepts_a_binary_mask_with_values_0_and_1(self, stenotic_mask):
        """The pipeline binarises with `> 0`, so a {0,1} mask works for detection."""
        binary = (stenotic_mask > 0).astype(np.uint8)
        fig = main.find_stenosis(binary * 255)[0]
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_works_on_each_bundled_example(self, data_dir):
        for name in ("1", "15"):
            mask = cv2.imread(str(data_dir / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(data_dir / "images" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            stenosis_img, overlay, paths_img = main.find_stenosis(mask, image)
            assert len(stenosis_img.axes) == 2
            assert len(overlay.axes) == 2


class TestMainEntryPoint:
    def test_shows_all_three_figures(self, monkeypatch):
        figures = [_FakeFigure(), _FakeFigure(), _FakeFigure()]
        monkeypatch.setattr(main, "find_stenosis", lambda mask, image: tuple(figures))
        main.main()
        assert all(f.shown for f in figures)

    def test_reads_mask_and_image_number_1(self, monkeypatch):
        seen = []
        real_imread = cv2.imread

        def spy(path, flags=None):
            seen.append(str(path))
            return real_imread(path, flags)

        monkeypatch.setattr(main.cv2, "imread", spy)
        monkeypatch.setattr(
            main, "find_stenosis",
            lambda mask, image: (_FakeFigure(), _FakeFigure(), _FakeFigure()),
        )
        main.main()
        assert any(p.endswith("masks/1.png") or p.endswith("masks\\1.png") for p in seen)
        assert any(p.endswith("images/1.png") or p.endswith("images\\1.png") for p in seen)

    def test_missing_files_raise_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(main.cv2, "imread", lambda *a, **k: None)
        with pytest.raises(FileNotFoundError, match="Could not read"):
            main.main()
