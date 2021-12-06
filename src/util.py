"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

def mi_converter(condition, invert=False):
    if not invert:
        if condition == 'BF':
            return 0
        if condition == 'RH':
            return 1
        if condition == 'LH':
            return 2
        if condition == 'TG':
            return 3
    else:
        if condition == 0:
            return "BF"
        if condition == 1:
            return "RH"
        if condition == 2:
            return "LH"
        if condition == 3:
            return 'TG'

def errp_converter(condition):
    if condition == 2:
        return "No error found"
    else: 
        return "Error detected"

