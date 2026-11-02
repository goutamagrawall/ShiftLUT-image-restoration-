"""Run with: python -m unittest discover -s tests -v"""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui_utils import SAMPLES, clean_sample, png_bytes, prepare_upload, sample_image, wipe_preview


class ImagePreparationTests(unittest.TestCase):
    def test_every_sample_is_available_in_each_restoration_mode(self):
        for sample in SAMPLES:
            clean = clean_sample(sample)
            for task in ('sr', 'denoising', 'deblocking'):
                with self.subTest(sample=sample, task=task):
                    raw = sample_image(task, sample)
                    prepared, original = prepare_upload(raw, task, None)
                    expected = (clean.width // 4, clean.height // 4) if task == 'sr' else clean.size
                    self.assertEqual(original, expected)
                    self.assertEqual(prepared.mode, 'RGB' if task == 'sr' else 'L')
                    self.assertEqual(raw, sample_image(task, sample))

    def test_large_input_resizes_explicitly_and_retains_aspect(self):
        image = Image.new('RGB', (1200, 600), '#725490')
        prepared, original = prepare_upload(png_bytes(image), 'sr', 256)
        self.assertEqual(original, (1200, 600))
        self.assertEqual(prepared.size, (256, 128))
        self.assertEqual(prepared.mode, 'RGB')
        with self.assertRaises(ValueError):
            prepare_upload(png_bytes(Image.new('RGB', (1200, 1200))), 'sr', None)

    def test_wipe_endpoints_and_grayscale_preparation(self):
        before = Image.new('RGB', (20, 10), 'red')
        after = Image.new('RGB', (80, 40), 'blue')
        self.assertEqual(wipe_preview(before, after, 0).getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(wipe_preview(before, after, 100).getpixel((0, 0)), (255, 0, 0))
        prepared, _ = prepare_upload(png_bytes(before), 'denoising', None)
        self.assertEqual(prepared.mode, 'L')


class InterfaceTests(unittest.TestCase):
    def assert_healthy(self, app):
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))

    def test_gallery_buttons_picker_and_sample_change_clear_result(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
        app.segmented_control(key='source').set_value('Try a sample').run()
        for sample in SAMPLES:
            app.button(key=f'sample_{sample}').click().run()
            self.assert_healthy(app)
            self.assertEqual(app.selectbox(key='selected_sample').value, sample)
        app.selectbox(key='selected_sample').set_value('portrait').run()
        app.button(key='restore').click().run()
        self.assert_healthy(app)
        result = app.session_state['restoration_result']
        with Image.open(io.BytesIO(sample_image('sr', 'portrait'))) as source:
            self.assertEqual(result['output_size'], (source.width * 4, source.height * 4))
        self.assertEqual(result['filename'], 'sample_portrait_sr_sr.png')
        app.button(key='sample_cat').click().run()
        self.assert_healthy(app)
        self.assertIsNone(app.session_state['restoration_result'])
        self.assertFalse(app.get('download_button'))
        app.segmented_control(key='task').set_value('Remove noise').run()
        self.assert_healthy(app)
        self.assertEqual(app.selectbox(key='selected_sample').value, 'cat')

    def test_sample_restoration_all_tasks_and_stale_result_clearing(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
        self.assert_healthy(app)
        self.assertTrue(app.button(key='restore').disabled)
        app.segmented_control(key='source').set_value('Try a sample').run()
        for label in ('Enlarge 4\u00d7', 'Remove noise', 'Clean up JPEG'):
            app.segmented_control(key='task').set_value(label).run()
            self.assertIsNone(app.session_state['restoration_result'])
            app.button(key='restore').click().run()
            self.assert_healthy(app)
            result = app.session_state['restoration_result']
            self.assertEqual(result['output_size'], (192, 128))
            self.assertTrue(app.get('download_button'))
        last_result = app.session_state['restoration_result']
        app.segmented_control(key='view').set_value('Wipe comparison').run()
        app.slider(key='split').set_value(70).run()
        self.assert_healthy(app)
        self.assertEqual(app.session_state['restoration_result']['after'], last_result['after'])
        self.assertEqual(app.session_state['restoration_result']['seconds'], last_result['seconds'])
        app.selectbox(key='processing_size').set_value('Quick preview \u00b7 256 px').run()
        self.assertIsNone(app.session_state['restoration_result'])
        self.assertFalse(app.get('download_button'))

    def test_uploaded_photo_upscales_original_dimensions_by_default(self):
        uploaded = io.BytesIO(png_bytes(Image.new('RGB', (300, 150), '#725490')))
        uploaded.name = 'portrait.png'
        with patch('streamlit.file_uploader', return_value=uploaded):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assertEqual(app.selectbox(key='processing_size').value, 'Original size')
            app.button(key='restore').click().run()
            self.assert_healthy(app)
            result = app.session_state['restoration_result']
            self.assertEqual(result['input_size'], (300, 150))
            with Image.open(io.BytesIO(result['after'])) as output:
                self.assertEqual(output.size, (1200, 600))

    def test_preview_warns_when_output_is_smaller_than_upload(self):
        uploaded = io.BytesIO(png_bytes(Image.new('RGB', (1200, 600))))
        uploaded.name = 'large_portrait.png'
        with patch('streamlit.file_uploader', return_value=uploaded):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assert_healthy(app)
            app.selectbox(key='processing_size').set_value('Quick preview \u00b7 256 px').run()
            self.assert_healthy(app)
            self.assertTrue(any('smaller than your uploaded photo' in warning.value for warning in app.warning))
            self.assertTrue(any('Output: 1024' in caption.value for caption in app.caption))

    def test_oversized_original_requires_explicit_preview_selection(self):
        uploaded = io.BytesIO(png_bytes(Image.new('RGB', (1200, 1200))))
        uploaded.name = 'large.png'
        with patch('streamlit.file_uploader', return_value=uploaded):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assertFalse(list(app.exception))
            self.assertTrue(app.error)
            self.assertTrue(app.button(key='restore').disabled)
            app.selectbox(key='processing_size').set_value('Larger preview \u00b7 512 px').run()
            self.assert_healthy(app)
            self.assertFalse(app.button(key='restore').disabled)

    def test_upload_processing_and_invalid_image_error(self):
        uploaded = io.BytesIO(sample_image('denoising'))
        uploaded.name = 'my noisy photo.png'
        with patch('streamlit.file_uploader', return_value=uploaded):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            app.segmented_control(key='task').set_value('Remove noise').run()
            app.button(key='restore').click().run()
            self.assert_healthy(app)
            self.assertEqual(app.session_state['restoration_result']['filename'], 'mynoisyphoto_denoising.png')
            with Image.open(io.BytesIO(app.session_state['restoration_result']['after'])) as output:
                self.assertEqual(output.mode, 'L')
                self.assertEqual(output.size, (192, 128))
        invalid = io.BytesIO(b'not an image')
        invalid.name = 'broken.png'
        with patch('streamlit.file_uploader', return_value=invalid):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assertFalse(list(app.exception))
            self.assertTrue(app.error)
            self.assertTrue(app.button(key='restore').disabled)


if __name__ == '__main__':
    unittest.main()
