class ReleasedVae:
    """
    Placeholder that replaces the VAE reference in pipeline modules after the VAE was released
    by the staged loading mode (release_text_encoder_after_cache). It holds no model weights, so
    the released VAE can be garbage collected; calling encode/decode raises a clear error instead
    of a confusing AttributeError, because encoding images without the VAE means the image cache
    is incomplete or was invalidated.
    """

    def encode(self, *args, **kwargs):
        raise RuntimeError(
            "The VAE was released after latent caching (release_text_encoder_after_cache), but "
            "something tried to encode an image. This means the image cache is incomplete or was "
            "invalidated (changed images, crops, resolutions, concepts or cache dir). Restart the "
            "training so the cache can be rebuilt, or disable release_text_encoder_after_cache."
        )

    def decode(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def __repr__(self):
        return "<ReleasedVae>"
