class ComponentNotLoadedError(RuntimeError):
    """
    Raised by a cache stage's before_cache_fun when the cache turns out to be incomplete but the
    component needed to complete it (text encoder / VAE) is not loaded. The staged loading mode
    (release_text_encoder_after_cache) uses this as a probe signal: it first warms the cache with
    the component deliberately unloaded; if nothing raises, the cache is fully valid and the
    component never has to be loaded at all. If this error is raised, the component is loaded and
    the cache stage is re-run.
    """
