import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import re
import string
from task3_preprocess import preprocess_text, identify_vn, load_data, yield_tokens, collate_ba