"""Third-party skill system for QitOS."""

from .bundled import (
    BundledSkillDiagnostic,
    BundledSkillDiagnosticCode,
    BundledSkillSnapshot,
)
from .cli import main as skill_cli_main
from .injector import (
    AutoSkillSelector,
    PromptContext,
    SkillInjector,
    SkillPromptBuilder,
)
from .integration import SkillMixin, SkilledAgent
from .loader import SkillInstaller, SkillLoader
from .manager import SkillManager
from .manifest import (
    InstalledSkill,
    SkillManifest,
    SkillPackage,
    parse_skill_package_dir,
    validate_skill_structure,
)
from .provider import (
    LocalSkillProvider,
    SkillDownload,
    SkillHubProvider,
    SkillProvider,
    SkillSearchResult,
)
from .registry import RegistryEntry, SkillRegistry, installed_to_entry

__all__ = [
    "BundledSkillDiagnostic",
    "BundledSkillDiagnosticCode",
    "BundledSkillSnapshot",
    "SkillManifest",
    "SkillPackage",
    "InstalledSkill",
    "SkillLoader",
    "SkillInstaller",
    "SkillManager",
    "SkillRegistry",
    "RegistryEntry",
    "SkillProvider",
    "SkillHubProvider",
    "LocalSkillProvider",
    "SkillSearchResult",
    "SkillDownload",
    "SkillMixin",
    "SkilledAgent",
    "SkillInjector",
    "SkillPromptBuilder",
    "AutoSkillSelector",
    "PromptContext",
    "parse_skill_package_dir",
    "installed_to_entry",
    "validate_skill_structure",
    "skill_cli_main",
]
