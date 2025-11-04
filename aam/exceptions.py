"""Custom exception classes for AAM (Attention-based Abundance Model).

This module provides exception classes specific to AAM operations, allowing
for better error handling and debugging throughout the codebase.
"""


class AAMError(Exception):
    """Base exception class for all AAM-related errors.
    
    All custom exceptions in this module inherit from this class, allowing
    callers to catch all AAM errors with a single exception type.
    
    Attributes:
        message: The error message describing what went wrong.
        context: Optional dictionary containing additional context about the error,
                 such as parameter names and values that caused the error.
    """
    
    def __init__(self, message: str, context: dict | None = None):
        """Initialize the exception.
        
        Args:
            message: A descriptive error message.
            context: Optional dictionary with additional context (e.g., 
                    parameter names, values, file paths).
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return a formatted error message with context if available."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ModelConfigurationError(AAMError):
    """Raised when model configuration is invalid or incompatible.
    
    This exception is used when model parameters conflict or are incompatible
    with each other. For example, when both taxonomy and tree are provided,
    or when neither is provided but one is required.
    
    Example:
        raise ModelConfigurationError(
            "Only taxonomy or UniFrac is supported, not both.",
            context={"taxonomy": taxonomy_path, "tree": tree_path}
        )
    """
    pass


class DataLoadError(AAMError):
    """Raised when data loading fails.
    
    This exception is used for errors during BIOM table loading, metadata
    file loading, or data validation failures.
    
    Example:
        raise DataLoadError(
            "Failed to load BIOM table",
            context={"file_path": table_path, "error": str(e)}
        )
    """
    pass


class ModelLoadError(AAMError):
    """Raised when model loading fails.
    
    This exception is used when loading a pre-trained base model fails,
    either due to file issues or model format incompatibilities.
    
    Example:
        raise ModelLoadError(
            "Failed to load base model",
            context={"model_path": model_path, "error": str(e)}
        )
    """
    pass


class TrainingError(AAMError):
    """Raised when model training fails.
    
    This exception is used for errors during the training process, such as
    training loop failures, callback errors, or optimization issues.
    
    Example:
        raise TrainingError(
            "Training failed during fold 2",
            context={"fold": 2, "epoch": 10, "error": str(e)}
        )
    """
    pass
