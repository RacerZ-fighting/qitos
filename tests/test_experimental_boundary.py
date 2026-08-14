"""Test deprecated product templates that remain outside the canonical tool API."""

import warnings


class TestSecurityAuditAgentShim:
    def test_import_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib
            import qitos.kit.agent.security_audit_agent as mod
            importlib.reload(mod)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1, "Expected at least one DeprecationWarning"
            msg = str(deprecation_warnings[0].message)
            assert "deprecated" in msg.lower()
            assert "security_audit_agent" in msg
