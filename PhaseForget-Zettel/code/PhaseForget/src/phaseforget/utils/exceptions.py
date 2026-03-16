"""
Custom exception hierarchy for PhaseForget-Zettel.
"""


class PhaseForgetError(Exception):
    """Base exception for all PhaseForget errors."""
    pass


class StorageError(PhaseForgetError):
    """Raised when a storage operation fails."""
    pass


class ColdTrackError(StorageError):
    """Raised when ChromaDB operations fail."""
    pass


class HotTrackError(StorageError):
    """Raised when SQLite operations fail."""
    pass


class LLMError(PhaseForgetError):
    """Raised when LLM generation or parsing fails."""
    pass


class RenormalizationError(PhaseForgetError):
    """Raised when the renormalization pipeline encounters an unrecoverable error."""
    pass


class CooldownViolationError(PhaseForgetError):
    """Raised when attempting to trigger renormalization on a cooling-down node."""
    pass
