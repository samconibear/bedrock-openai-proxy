class GatewayError(Exception):
    status_code: int = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AuthError(GatewayError):
    status_code = 401


class ValidationError(GatewayError):
    status_code = 400


class ProviderError(GatewayError):
    status_code = 502


class RateLimitError(GatewayError):
    status_code = 429
