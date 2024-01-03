import unittest

from lib import Router, Route
from lib.transformers_engine import SentenceTransformersEngine


class TestRouter(unittest.TestCase):

    def test_transformer_router(self):
        router = Router(
            engine=SentenceTransformersEngine(
                model_name='all-distilroberta-v1',
                threshold=0.3,
            ),
            routes=[
                Route(
                    name='upscale',
                    sentences=[
                        'upscale the image',
                        'increase resolution',
                        'I want a larger image',
                        'increase the pixel count',
                        'increase the size of the image',
                        'increase the resolution of the image',
                        'I think this is too small',
                    ]
                ),
                Route(
                    name='edit',
                    sentences=[
                        'edit image',
                        'rotate image',
                        'flip image',
                        'resize image',
                        'adjust contrast',
                        'adjust saturation',
                        'adjust brightness',
                        'change colors',
                        'color balance',
                        'change image format',
                        'change dimensions',
                        'change size',
                        'crop image',
                    ]
                )
            ]
        )

        assert router.match('make this big') == 'upscale'
        assert router.match('I can has big image?') == 'upscale'
        assert router.match('enlarge!') == 'upscale'
        assert router.match('is it possible to make this better in terms of graphical quality') == 'upscale'

        assert router.match('i said i want black and white!!!') == 'edit'
        assert router.match('cut it in half') == 'edit'