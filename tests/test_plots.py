import matplotlib.figure
import numpy as np
import pytest

from helpers import run_detection
from plots import draw_stenosis, plot_overlay, plot_stenosis, visualize_paths


@pytest.fixture
def scene(stenotic_mask):
    """(mask, image, skeleton, paths, detections) for a stenotic vessel."""
    skeleton, paths, _, detections = run_detection(stenotic_mask)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, stenotic_mask.shape, dtype=np.uint8)
    return stenotic_mask, image, skeleton, paths, detections


def _rgb(mask):
    return np.stack([mask] * 3, axis=-1).copy()


#***************************************************************
# draw_stenosis
#***************************************************************

class TestDrawStenosis:
    def test_no_detections_leaves_the_image_unchanged(self, scene):
        mask = scene[0]
        vis = _rgb(mask)
        before = vis.copy()
        out = draw_stenosis(vis, [])
        assert np.array_equal(out, before)

    def test_detections_change_the_image(self, scene):
        mask, *_, detections = scene
        vis = _rgb(mask)
        before = vis.copy()
        out = draw_stenosis(vis, detections)
        assert not np.array_equal(out, before)

    def test_contour_is_drawn_in_blue_channel_colour(self, scene):
        """Contours use (255, 0, 0): a pixel that was black becomes pure red."""
        mask, *_, detections = scene
        out = draw_stenosis(_rgb(mask), detections, draw_text=False)
        red = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 0)
        assert red.any()

    def test_text_is_optional(self, scene):
        mask, *_, detections = scene
        with_text = draw_stenosis(_rgb(mask), detections, draw_text=True)
        without = draw_stenosis(_rgb(mask), detections, draw_text=False)
        assert not np.array_equal(with_text, without)

    def test_returns_the_same_array_it_was_given(self, scene):
        mask, *_, detections = scene
        vis = _rgb(mask)
        assert draw_stenosis(vis, detections) is vis

    def test_detection_with_out_of_range_idx_is_skipped(self, scene):
        mask, *_, detections = scene
        bad = dict(detections[0], idx=len(detections[0]["path"]) + 5)
        vis = _rgb(mask)
        before = vis.copy()
        draw_stenosis(vis, [bad])
        assert np.array_equal(vis, before)

    @pytest.mark.parametrize(
        "g_set, l_set, label",
        [({7}, {7}, "B"), ({7}, set(), "G"), (set(), {7}, "L")],
    )
    def test_label_reflects_which_detector_found_the_lesion(
        self, scene, monkeypatch, g_set, l_set, label
    ):
        import plots

        mask, *_, detections = scene
        det = dict(detections[0], orig_idx=7, g_set=g_set, l_set=l_set, severity=0.5)

        captured = []
        monkeypatch.setattr(
            plots.cv2, "putText",
            lambda img, text, *a, **k: captured.append(text),
        )
        draw_stenosis(_rgb(mask), [det])
        assert captured == [f"{label}:50%"]

    def test_severity_is_shown_as_a_percentage(self, scene, monkeypatch):
        import plots

        mask, *_, detections = scene
        det = dict(detections[0], severity=0.64)

        captured = []
        monkeypatch.setattr(
            plots.cv2, "putText",
            lambda img, text, *a, **k: captured.append(text),
        )
        draw_stenosis(_rgb(mask), [det])
        assert captured[0].endswith(":64%")


#***************************************************************
# plot_stenosis
#***************************************************************

class TestPlotStenosis:
    def test_mask_only_returns_a_figure_with_one_axes(self, scene):
        mask, _, _, _, detections = scene
        fig = plot_stenosis(mask, detections)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1

    def test_with_image_returns_two_panels(self, scene):
        mask, image, _, _, detections = scene
        fig = plot_stenosis(mask, detections, image)
        assert len(fig.axes) == 2

    def test_works_without_any_detections(self, scene):
        mask = scene[0]
        fig = plot_stenosis(mask, [])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_titles(self, scene):
        mask, image, _, _, detections = scene
        assert plot_stenosis(mask, detections).axes[0].get_title() == "Detected Stenosis"
        fig = plot_stenosis(mask, detections, image)
        assert [a.get_title() for a in fig.axes] == ["Image + Stenosis", "Mask + Stenosis"]

    def test_input_mask_is_not_modified(self, scene):
        mask, _, _, _, detections = scene
        before = mask.copy()
        plot_stenosis(mask, detections)
        assert np.array_equal(mask, before)


#***************************************************************
# plot_overlay
#***************************************************************

class TestPlotOverlay:
    def test_mask_only(self, scene):
        mask, _, skeleton, _, _ = scene
        fig = plot_overlay(mask, skeleton)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1

    def test_with_image_returns_two_panels(self, scene):
        mask, image, skeleton, _, _ = scene
        fig = plot_overlay(mask, skeleton, image)
        assert len(fig.axes) == 2

    def test_skeleton_pixels_are_painted_red(self, scene):
        mask, _, skeleton, _, _ = scene
        fig = plot_overlay(mask, skeleton)
        shown = np.asarray(fig.axes[0].images[0].get_array())
        sk = skeleton > 0
        assert (shown[sk] == [255, 0, 0]).all()

    def test_pixels_off_the_skeleton_are_unchanged(self, scene):
        mask, _, skeleton, _, _ = scene
        fig = plot_overlay(mask, skeleton)
        shown = np.asarray(fig.axes[0].images[0].get_array())
        off = skeleton == 0
        assert np.array_equal(shown[off][:, 0], mask[off])

    def test_accepts_colour_mask_and_image(self, scene):
        mask, image, skeleton, _, _ = scene
        fig = plot_overlay(_rgb(mask), skeleton, _rgb(image))
        assert len(fig.axes) == 2


#***************************************************************
# visualize_paths
#***************************************************************

class TestVisualizePaths:
    def test_mask_only_has_three_panels(self, scene):
        mask, _, skeleton, paths, _ = scene
        plt_module = visualize_paths(mask, skeleton, paths)
        assert len(plt_module.gcf().axes) == 3

    def test_with_image_has_four_panels(self, scene):
        mask, image, skeleton, paths, _ = scene
        plt_module = visualize_paths(mask, skeleton, paths, image)
        assert len(plt_module.gcf().axes) == 4

    def test_returns_the_pyplot_module_so_show_can_be_called(self, scene):
        mask, _, skeleton, paths, _ = scene
        plt_module = visualize_paths(mask, skeleton, paths)
        assert hasattr(plt_module, "show") and hasattr(plt_module, "gcf")

    def test_every_path_pixel_is_drawn_on_the_canvas(self, scene):
        mask, _, skeleton, paths, _ = scene
        plt_module = visualize_paths(mask, skeleton, paths)
        canvas = np.asarray(plt_module.gcf().axes[2].images[0].get_array())
        for path in paths:
            for x, y in path:
                assert canvas[y, x].any()

    def test_no_paths_gives_a_black_canvas(self, scene):
        mask, _, skeleton, _, _ = scene
        plt_module = visualize_paths(mask, skeleton, [])
        canvas = np.asarray(plt_module.gcf().axes[2].images[0].get_array())
        assert not canvas.any()
