class OmniInputValidationError(Exception):
    """Raised when a pipeline worker rejects a request due to invalid input.

    Caught explicitly by API-layer handlers to return HTTP 400.
    """
