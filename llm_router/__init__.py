from typing import Optional


class Engine:
    def encode(self, routes):
        raise NotImplementedError()

    def match(self, query) -> Optional[str]:
        raise NotImplementedError()


class Route:
    def __init__(self, name, sentences):
        self.sentences = sentences
        self.name = name


class Router:
    def __init__(self, routes, engine):
        self.engine = engine
        self.engine.encode(routes)

    def match(self, query) -> Optional[str]:
        return self.engine.match(query)
