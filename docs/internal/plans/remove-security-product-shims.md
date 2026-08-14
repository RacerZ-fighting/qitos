# Remove security product shims from the default kit

## Problem

Importing `qitos.kit` currently loads the experimental security-research package through
the generic `qitos.kit.toolset` initializer. The same capability is also exposed through
three forwarding packages and an unused product-level `SecurityAuditAgent` template.
This violates the documented explicit opt-in boundary and makes import behavior depend
on test order.

## Changes

- Keep `qitos.kit.tool.experimental.security_research` as the single explicit owner.
- Remove the deprecated product Agent template and security forwarding packages.
- Remove generic `toolset` exports and tests that exercise only those forwarding paths.
- Preserve behavior tests that import and use the experimental owner directly.
- Update public maintenance notes and the core-governance record.

## Acceptance

- Importing `qitos` or `qitos.kit` does not load experimental security modules.
- Explicit security-research imports retain their existing behavior.
- The QitOS Python 3.11 test, static, and packaging checks pass.
