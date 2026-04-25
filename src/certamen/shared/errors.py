class CertamenError(Exception):
    def __init__(self, message: str, *args: object) -> None:
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(CertamenError):
    pass


class FileSystemError(CertamenError):
    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        *args: object,
    ) -> None:
        self.file_path = file_path
        enhanced_message = message

        if file_path:
            enhanced_message = f"[{file_path}] {enhanced_message}"

        super().__init__(enhanced_message, *args)
