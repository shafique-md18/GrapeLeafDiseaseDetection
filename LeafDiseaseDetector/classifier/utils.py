from django.utils.text import slugify
import random
import string
import functools
import os


def random_string_generator(size=10, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def unique_file_name(path=None):
    file_name = random_string_generator(size=10) + '.jpg'
    # to-do: check if file name is unique
    return file_name

