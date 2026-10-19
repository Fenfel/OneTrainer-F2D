class ReleasedTextEncoder:
    """
    Placeholder that replaces the text encoder reference in pipeline modules after the text
    encoder was released (release_text_encoder_after_cache). It holds no model weights, so the
    released encoder can be garbage collected; calling it raises a clear error instead of a
    confusing AttributeError, because encoding text without the encoder means the text cache
    is incomplete or was invalidated.
    """

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "The text encoder was released after text caching (release_text_encoder_after_cache), "
            "but something tried to encode text. This means the text cache is incomplete or was "
            "invalidated (changed captions, text variations, concepts, cache dir or text encoder). "
            "Restart the training so the cache can be rebuilt, or disable release_text_encoder_after_cache."
        )

    def __repr__(self):
        return "<ReleasedTextEncoder>"
