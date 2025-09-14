class DialecticaError(Exception):
    """Base error for Dialectica."""


class ProviderError(DialecticaError):
    pass


class PromptError(DialecticaError):
    pass


class ArtifactError(DialecticaError):
    pass

