"""Exception hierarchy. Every VEIL-X failure raises a subclass of :class:`VeilxError`.

Messages must never contain image content, embeddings or personal names; identifiers
(dataset ids, image ids, identity ids) are allowed because they are pseudonymous keys.
"""

from __future__ import annotations


class VeilxError(Exception):
    """Base class for all VEIL-X errors."""


class ConfigError(VeilxError):
    """A configuration file is missing, malformed, or violates the research protocol."""


class ManifestError(VeilxError):
    """A dataset or pair manifest violates its schema or integrity rules."""


class LeakageError(VeilxError):
    """Data from one partition (identity split or held-out family) leaked into another."""


class ProtocolViolationError(VeilxError):
    """An operation would break the pre-registered protocol (e.g. unflagged test access)."""


class DataAccessError(VeilxError):
    """A dataset is not approved for the requested use (licence, access or ethics status)."""


class InsufficientSamplesError(VeilxError):
    """A metric or operating point is requested that the available samples cannot support."""


class ModelLoadError(VeilxError):
    """A checkpoint does not match its pinned hash or its architecture (missing, unexpected
    or mis-shaped tensors); loading never proceeds with a partial match."""


class InfeasibleRequestError(VeilxError):
    """A requested construction (e.g. N independent pairs) cannot be satisfied by the data."""
