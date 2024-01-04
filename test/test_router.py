import os
import unittest

from llm_router import Router, Route
from llm_router.chroma import SentenceTransformer, OpenAI


class TestRouter(unittest.TestCase):

    def test_transformer_router(self):
        router = Router(
            engine=SentenceTransformer(
                threshold=0.4,
                model_name='all-distilroberta-v1',
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

        self.assertEquals(router.match('make this big'), 'upscale')
        self.assertEquals(router.match('I can has big image?'), 'upscale')
        self.assertEquals(router.match('enlarge!'), 'upscale')
        self.assertEquals(router.match('is it possible to make this better in terms of graphical quality'), 'upscale')

        self.assertEquals(router.match('i said i want black and white!!!'), 'edit')
        self.assertEquals(router.match('cut it in half'), 'edit')

    def test_openai_router(self):
        router = Router(
            engine=OpenAI(
                cache_path="./test_openai_cache",
                api_key=os.environ.get('OPENAI_API_KEY'),
            ),
            routes=[
                Route(
                    name='upscale',
                    sentences=[
                        'upscale the image',
                        'increase image resolution',
                        'increase pixel count',
                        'increase the size of an image',
                        'increase the resolution of the image',
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
                        'change or adjust colors, for instance make entire image use a specific color palette, make images black and white, etc',
                        'color balance',
                        'change image format',
                        'change dimensions',
                        'change size',
                        'crop image',
                    ]
                )
            ]
        )
        self.assertEquals(router.match('I can has big image?'), 'upscale')
        self.assertEquals(router.match('enlarge!'), 'upscale')
        self.assertEquals(router.match('is it possible to make this better in terms of graphical quality'), 'upscale')

        self.assertEquals(router.match('i said i want black and white!!!'), 'edit')
        self.assertEquals(router.match('cut it in half'), 'edit')
