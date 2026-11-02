"""Local image upload, restoration, comparison, and PNG download."""
import hashlib
from io import BytesIO
import threading
import time

from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from restore import load_model, restore_image
from ui_utils import SAMPLES, clean_sample, download_name, png_bytes, prepare_upload, preview, sample_image, wipe_preview

st.set_page_config(page_title='ShiftLUT | Image restoration', page_icon=':material/photo_filter:', layout='wide')
st.session_state.setdefault('restoration_result', None)

TASKS = {
    'Enlarge 4\u00d7': ('sr', 'Give a small image more pixels. Color is preserved.'),
    'Remove noise': ('denoising', 'Reduce image noise with the supplied sigma 15 model. The result is grayscale.'),
    'Clean up JPEG': ('deblocking', 'Reduce compression artifacts with the supplied quality 10 model. The result is grayscale.'),
}
SIZES = {'Quick preview \u00b7 256 px': 256, 'Larger preview \u00b7 512 px': 512, 'Original size': None}
MODELS = {'Smallest': 0, 'Balanced': 1, 'Full model': 7}


@st.cache_resource(max_entries=5, show_spinner=False)
def cached_model(task, stacks):
    torch.set_num_threads(1)
    module, model = load_model(task, stacks, 'cpu')
    return module, model, threading.Lock()


@st.cache_data(max_entries=15, show_spinner=False)
def cached_sample(task, sample):
    return sample_image(task, sample)


@st.cache_data(max_entries=5, show_spinner=False)
def cached_sample_thumbnail(sample):
    return png_bytes(preview(clean_sample(sample), (160, 100)))


def select_sample(sample):
    st.session_state.selected_sample = sample


st.caption(':material/photo_filter:  SHIFTLUT / IMAGE RESTORATION')
st.title('A clearer view, one image at a time.')
st.markdown('Upload a photo, choose a restoration, and see the difference.')
with st.container(horizontal=True):
    st.badge('Runs on this computer', icon=':material/computer:', color='green')
    st.caption('Images are processed locally, without a cloud model service.')
st.space('small')

controls, workspace = st.columns([1, 2], gap='large')
with controls:
    with st.container(border=True):
        st.subheader('1. Choose an image')
        source = st.segmented_control('Image source', ['Upload image', 'Try a sample'], default='Upload image', required=True, key='source', label_visibility='collapsed')
        uploaded = None
        sample = None
        if source == 'Upload image':
            uploaded = st.file_uploader('Attach your image', type=['png', 'jpg', 'jpeg', 'webp', 'bmp'], max_upload_size=20, key='upload', help='Drag a file here or browse your computer. Maximum file size: 20 MB.')
        else:
            sample = st.selectbox('Sample image', list(SAMPLES), format_func=lambda value: SAMPLES[value][0], key='selected_sample')
            st.caption(SAMPLES[sample][2])
            st.caption('Samples are made low resolution, noisy, or JPEG-compressed to match your restoration choice.')
        st.subheader('2. Choose a restoration')
        choice = st.segmented_control('Restoration', list(TASKS), default='Enlarge 4\u00d7', required=True, key='task', width='stretch', label_visibility='collapsed')
        task, explanation = TASKS[choice]
        st.caption(explanation)
        size_choice = st.selectbox('Processing size', list(SIZES), index=2, key='processing_size', help='Original size keeps all uploaded pixels (up to 1 megapixel). Preview sizes shrink larger uploads before restoration and also reduce the downloaded output dimensions.')
        stacks = 7
        if task == 'sr':
            model_choice = st.selectbox('Restoration model', list(MODELS), index=1, key='sr_model', help='All three models enlarge by 4\u00d7. Smaller models are faster; quality depends on the image.')
            stacks = MODELS[model_choice]
        st.caption('Designed for low resolution, noise, and JPEG artifacts. Motion blur and defocus removal are not supplied tasks.')

        raw = cached_sample(task, sample) if source == 'Try a sample' else (uploaded.getvalue() if uploaded is not None else None)
        name = f'sample_{sample}_{task}.png' if source == 'Try a sample' else (uploaded.name if uploaded is not None else 'image.png')
        prepared = None
        original_size = None
        if raw is not None:
            try:
                prepared, original_size = prepare_upload(raw, task, SIZES[size_choice])
            except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as exc:
                st.error(f'Could not prepare this image: {exc}')
        fingerprint = hashlib.sha256(raw).hexdigest() + f'/{name}/{task}/{stacks}/{SIZES[size_choice]}' if prepared is not None else None
        previous = st.session_state.restoration_result
        if previous and previous['fingerprint'] != fingerprint:
            st.session_state.restoration_result = None
        if prepared is not None:
            scale = 4 if task == 'sr' else 1
            st.caption(f'Uploaded: {original_size[0]} \u00d7 {original_size[1]}  \u2192  Processing: {prepared.width} \u00d7 {prepared.height}  \u2192  Output: {prepared.width * scale} \u00d7 {prepared.height * scale}')
            if prepared.size != original_size:
                st.warning(f'This preview setting downsizes your photo to {prepared.width} \u00d7 {prepared.height} before restoration. The download uses this reduced input, too. Choose Original size to process all uploaded pixels (up to 1 megapixel).')
                if prepared.width * scale < original_size[0] or prepared.height * scale < original_size[1]:
                    st.warning('The final output will be smaller than your uploaded photo, even after restoration.')
        run = st.button('Restore image', type='primary', icon=':material/auto_fix_high:', width='stretch', disabled=prepared is None, key='restore')

with workspace:
    if source == 'Try a sample':
        st.subheader('Sample gallery')
        for column, sample_id in zip(st.columns(len(SAMPLES)), SAMPLES):
            with column:
                st.image(cached_sample_thumbnail(sample_id), width='stretch', alt=f'{SAMPLES[sample_id][0]} sample thumbnail')
                st.button(SAMPLES[sample_id][0], key=f'sample_{sample_id}', type='primary' if sample_id == sample else 'secondary', width='stretch', on_click=select_sample, args=(sample_id,))
        st.caption('Choose a thumbnail button or use the sample picker. Thumbnails show the clean sources; the input below matches your restoration mode.')
    st.subheader('3. Preview your result')
    result_slot = st.container()

if run and prepared is not None:
    st.session_state.restoration_result = None
    try:
        with result_slot, st.spinner('Restoring your image on this computer\u2026', show_time=True):
            module, model, lock = cached_model(task, stacks)
            started = time.perf_counter()
            with lock:
                restored = restore_image(prepared, task, module, model)
            elapsed = time.perf_counter() - started
            st.session_state.restoration_result = {
                'fingerprint': fingerprint,
                'before': png_bytes(prepared),
                'after': png_bytes(restored),
                'input_size': prepared.size,
                'output_size': restored.size,
                'seconds': elapsed,
                'filename': download_name(name, task),
            }
    except (OSError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
        with result_slot:
            st.error(f'Restoration could not finish: {exc}. Try a smaller processing size.')

with result_slot:
    result = st.session_state.restoration_result
    if result:
        st.success('Your restored image is ready.', icon=':material/check_circle:')
        display = st.segmented_control('Comparison view', ['Side by side', 'Wipe comparison'], default='Side by side', required=True, key='view', label_visibility='collapsed')
        with Image.open(BytesIO(result['before'])) as image:
            before = image.copy()
        with Image.open(BytesIO(result['after'])) as image:
            after = image.copy()
        if display == 'Side by side':
            for column, label, image in zip(st.columns(2, border=True), ['Before', 'After'], [before, after]):
                with column:
                    st.markdown(f'**{label}**')
                    if label == 'Before':
                        st.caption('Image used for processing')
                    st.image(preview(image), width='content', alt=f'{label} restoration preview')
                    st.caption(f'{image.width} \u00d7 {image.height} pixels')
        else:
            split = st.slider('Comparison position', 0, 100, 50, key='split', format='%d%%')
            st.image(wipe_preview(before, after, split), width='content', alt='Wipe comparison with the input on the left and the restored result on the right')
            st.caption('Before on the left \u00b7 After on the right. Both are shown at the same display size.')
        st.caption(f'Restored in {result["seconds"]:.2f} seconds on CPU \u00b7 Previews fit the screen. Download: {after.width} \u00d7 {after.height} pixels.')
        st.download_button('Download restored image', data=result['after'], file_name=result['filename'], mime='image/png', icon=':material/download:', type='primary', width='stretch', on_click='ignore', key='download')
    elif prepared is not None:
        with st.container(border=True, horizontal_alignment='center'):
            st.image(preview(prepared), width='content', alt='Input image prepared for the selected restoration')
            st.caption('Your input image. Select Restore image to generate the result.')
    else:
        with st.container(border=True, height=390, vertical_alignment='center', horizontal_alignment='center'):
            st.subheader(':material/add_photo_alternate:  Your image goes here')
            st.markdown('Attach a photo on the left, or choose **Try a sample**.')
            st.caption('Compare before and after, then download a restored PNG.')
