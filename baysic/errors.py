"""Defines error types for 'clean' control flow."""

from click import Group
from baysic.lattice import LatticeModel
from pymatgen.core import Composition


class BaysicError(Exception):
    """Base class for all Baysic-specific exceptions."""
    pass

class StructureGenerationError(BaysicError):
    """Base error class for failure to generate structures. If coordinates aren't 
    found, CoordinateGenerationFailed is raised instead."""
    def __init__(self, comp: Composition, lattice_model: LatticeModel, groups: list[Group], message):
        self.comp = comp
        self.lattice = lattice_model
        self.groups = groups  
        self.message = message

class WyckoffAssignmentImpossible(StructureGenerationError):
    """Indicates that no Wyckoff assignment exists for the specified composition and groups."""
    pass

class WyckoffAssignmentFailed(StructureGenerationError):
    """Indicates that Wyckoff assignments exist but were not found."""
    pass

class CoordinateGenerationFailed(BaysicError):
    """Indicates that coordinate generation failed."""
    def __init__(self, log_info: dict, message):
        self.log_info = log_info
        self.message = message


class WyckoffAssignmentCacheError(BaysicError):
    """Indicates a composition was found that exceeds the precompute limit."""
    pass