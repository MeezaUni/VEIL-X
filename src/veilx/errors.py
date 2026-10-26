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
