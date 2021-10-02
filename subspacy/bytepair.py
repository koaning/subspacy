from typing import List, cast

from bpemb import BPEmb
from thinc.model import Model
from thinc.types import Floats2d, Ragged
from spacy.tokens import Doc
from spacy.util import registry


@registry.architectures("subspacy.BytePairEmbeddings.v1")
def BytePairEmbedding(vocab_size: int, dim: int, language: str) -> Model[List[Doc], List[Floats2d]]:
    module = BPEmb(lang=language, vs=vocab_size, dim=dim)
    return Model(
        name=f"bytepair_{language}_{vocab_size}_{dim}",
        forward=forward,
        init=init,
        attrs={"bpemb": module}
    )


def forward(model: Model[List[Doc], Ragged], docs: List[Doc]):
    module = model.attrs["bpemb"]
    vectors = cast(Floats2d, [[module.embed(t.text) for t in doc] for doc in docs])
    output = Ragged(
        vectors, model.ops.asarray([len(doc) for doc in docs], dtype="i")
    )
    return output


def init(model):
    model.set_dim("nO", model.attrs["bpemb"].dim)
