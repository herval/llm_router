# LLM Router

A fast semantic router using Embeddings.

LLM Router lets you define "routes" - sets of sentences or keywords with similar semantics. You can then use these routes to route a user's input to the appropriate LLM (or other model) for a response, leading to faster inference and better results. It's a faster and more efficient alternative to using a single LLM for all responses or letting a single LLM choose function calls.

It can work entirely locally using [sentence_transformers](https://huggingface.co/sentence-transformers) or using OpenAI's API.

Important: results WILL vary between openai and the sbert/sentence-transformer model you pick. You'll need to experiment to find the best model for your use case, as well as adjusting the threshold for your needs.

## But why?
This code was created as part of Clio AI (my most recent startup) - the idea then was to allow selecting the best "agent" to handle a certain conversation based on the semantics of what the user is saying, which allowed us to use "lesser" models like GPT-3.5 instead of GPT-4 to respond with the appropriate set of system prompt, function calls, etc. In some cases, it allows you to skip LLMs altogether, saving time and $$$.


## Usage

Install with:
```
pip install llm-router
```

To use the OpenAI API, you'll need an API Key and the `openai` pip package (`pip install openai`).

To use a Sentence Transformers model, you'll need the `sentence_transformers` pip package (`pip install sentence_transformers`). No API key is required, model weights will be downloaded from huggingface on first use.

By default, the Router will always match one of the routes. If you'd like to allow the user to say something that doesn't match any of the routes, you can set the `threshold` value when initializing the Router. This will return `None` if no routes match with a certain percentage (0 to 1.0).


Define routes in code and use the router like so:

```python
from llm_router import Router, Route
from llm_router.chroma import SentenceTransformer

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


user_message = 'I want to increase the resolution of the image, is that possible?'

if router.match(user_message) == 'upscale':
    handle_upscale()
elif router.match(user_message) == 'edit':
    handle_edit()
else:
    print('Sorry, I don\'t understand.')
```


## References

- [Pretrained Sentence Transformer models](https://www.sbert.net/docs/pretrained_models.html) - `all-mpnet-base-v2` is recommended for most use casesm but you can also use `all-distilroberta-v1` for a smaller model that's faster. I haven't tested other models yet.



## Future enhancements (PRs welcome!)

- Support for other embedding APIs (cohere, vertexai, etc)
- Storing weights to prevent reprocessing every time
- Add a finetuning script to create routes from a dataset and refine sentences