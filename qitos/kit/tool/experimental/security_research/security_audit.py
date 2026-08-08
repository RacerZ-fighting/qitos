"""Curated codebase security audit tools for developer-owned repositories."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qitos.core.function_tool_decorator import function_tool
from qitos.core.tool import ToolPermission

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except Exception:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


LANGUAGE_EXTENSIONS: Dict[str, set[str]] = {
    "python": {".py"},
    "javascript": {".js", ".cjs", ".mjs"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "go": {".go"},
    "ruby": {".rb"},
    "php": {".php"},
    "rust": {".rs"},
    "kotlin": {".kt"},
    "shell": {".sh", ".bash", ".zsh"},
    "yaml": {".yml", ".yaml"},
    "json": {".json"},
    "docker": {".dockerfile"},
}

MANIFEST_FILES = {
    "requirements.txt": "pip",
    "pyproject.toml": "python",
    "poetry.lock": "python-lock",
    "Pipfile": "pipenv",
    "Pipfile.lock": "pipenv-lock",
    "package.json": "npm",
    "package-lock.json": "npm-lock",
    "pnpm-lock.yaml": "pnpm-lock",
    "yarn.lock": "yarn-lock",
    "Cargo.toml": "cargo",
    "Cargo.lock": "cargo-lock",
    "go.mod": "gomod",
    "go.sum": "gomod-lock",
    "Gemfile": "bundler",
    "Gemfile.lock": "bundler-lock",
    "pom.xml": "maven",
}

SECURITY_RELEVANT_NAMES = (
    ".env",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    "kustomization.yaml",
    "Chart.yaml",
)

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".turbo",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
}

CODE_FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".php",
    ".rs",
    ".kt",
    ".sh",
}

TEXT_FILE_EXTENSIONS = CODE_FILE_EXTENSIONS | {
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".yaml",
    ".yml",
    ".xml",
    ".md",
    ".txt",
    ".properties",
    ".dockerfile",
}

PLACEHOLDER_SECRET_VALUES = {
    "changeme",
    "change-me",
    "example",
    "example-key",
    "test",
    "test-key",
    "dummy",
    "sample",
    "your_api_key",
    "xxx",
    "placeholder",
}


def _resolve_workspace_path(root_dir: str, path: str = ".") -> Path:
    root = Path(root_dir).expanduser().resolve()
    target = (root / (path or ".")).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f"Access denied: '{path}' is outside workspace '{root}'")
    return target


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = float(len(value))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


class SecurityAuditToolSet:
    """Atomic security-audit tools for developer-owned code repositories."""

    name = "security_audit"
    version = "1"

    def __init__(
        self,
        workspace_root: str = ".",
        *,
        include_external: bool = False,
        external_timeout: int = 120,
        max_matches: int = 200,
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        self.include_external = bool(include_external)
        self.external_timeout = int(external_timeout)
        self.max_matches = int(max_matches)
        self._session_cache: Dict[str, Dict[str, Any]] = {}

    def setup(self, context: Dict[str, Any]) -> None:
        _ = context

    def teardown(self, context: Dict[str, Any]) -> None:
        _ = context

    def tools(self) -> List[Any]:
        items: List[Any] = [
            self.audit_inventory,
            self.audit_entrypoints,
            self.audit_sink_scan,
            self.audit_secret_scan,
            self.audit_config_scan,
            self.audit_dependency_inventory,
            self.audit_notes_scan,
            self.audit_hotspots,
        ]
        if self.include_external:
            items.append(self.audit_dependency_audit)
        return items

    def _repo_root(self) -> Path:
        return _resolve_workspace_path(self.workspace_root)

    def _iter_repo_files(self) -> List[Path]:
        files: List[Path] = []
        for root, dirs, names in os.walk(self._repo_root()):
            dirs[:] = [
                d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".cache")
            ]
            for name in names:
                files.append(Path(root) / name)
        return files

    def _read_text(self, path: Path, max_bytes: int = 400_000) -> Optional[str]:
        try:
            raw = path.read_bytes()
        except Exception:
            return None
        if b"\x00" in raw:
            return None
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="ignore")

    def _relative(self, path: Path) -> str:
        return os.path.relpath(path, self.workspace_root)

    def _store(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._session_cache[key] = payload
        return payload

    def _code_files(self) -> List[Path]:
        files: List[Path] = []
        for path in self._iter_repo_files():
            suffix = path.suffix.lower()
            if suffix in CODE_FILE_EXTENSIONS or path.name in {"Dockerfile"}:
                files.append(path)
        return files

    def _text_files(self) -> List[Path]:
        files: List[Path] = []
        for path in self._iter_repo_files():
            suffix = path.suffix.lower()
            if (
                suffix in TEXT_FILE_EXTENSIONS
                or path.name in MANIFEST_FILES
                or path.name in SECURITY_RELEVANT_NAMES
            ):
                files.append(path)
        return files

    def _build_finding(
        self,
        *,
        title: str,
        category: str,
        severity: str,
        confidence: float,
        file: str,
        line: int,
        evidence: str,
        rationale: str,
        recommendation: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "title": title,
            "category": category,
            "severity": severity,
            "confidence": round(float(confidence), 2),
            "file": file,
            "line": int(line),
            "evidence": evidence,
            "rationale": rationale,
            "recommendation": recommendation,
            "tags": list(tags or []),
        }

    def _summarize_findings(
        self, findings: Sequence[Dict[str, Any]], heading: str
    ) -> str:
        if not findings:
            return f"{heading}: no high-signal candidates found."
        lines = [f"{heading}: {len(findings)} candidate(s)."]
        for item in findings[: min(5, len(findings))]:
            lines.append(
                f"- {item['severity']} {item['category']} at {item['file']}:{item['line']} "
                f"(confidence={item['confidence']})"
            )
        return "\n".join(lines)

    def _framework_hints(self, files: List[Path]) -> List[str]:
        hints: set[str] = set()
        for path in files:
            if path.name == "package.json":
                text = self._read_text(path) or ""
                for needle, hint in (
                    ('"express"', "express"),
                    ('"next"', "next.js"),
                    ('"react"', "react"),
                    ('"fastify"', "fastify"),
                    ('"nestjs"', "nestjs"),
                ):
                    if needle in text:
                        hints.add(hint)
            elif path.name == "requirements.txt":
                text = self._read_text(path) or ""
                for needle, hint in (
                    ("flask", "flask"),
                    ("fastapi", "fastapi"),
                    ("django", "django"),
                    ("requests", "requests"),
                ):
                    if needle in text.lower():
                        hints.add(hint)
            elif path.name == "pyproject.toml":
                text = (self._read_text(path) or "").lower()
                for needle, hint in (
                    ("fastapi", "fastapi"),
                    ("django", "django"),
                    ("flask", "flask"),
                    ("sqlalchemy", "sqlalchemy"),
                ):
                    if needle in text:
                        hints.add(hint)
            elif path.name == "Dockerfile":
                hints.add("docker")
        return sorted(hints)

    def _scan_entrypoints(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        patterns = [
            (
                "http_route",
                re.compile(r"@\s*(app|router|bp)\.(get|post|put|delete|patch|route)\b"),
            ),
            (
                "http_route",
                re.compile(r"\b(app|router)\.(get|post|put|delete|patch|use)\s*\("),
            ),
            ("framework_route", re.compile(r"\b(path|re_path)\s*\(")),
            (
                "rpc_handler",
                re.compile(r"\b(grpc|rpc|GraphQL|resolver)\b", re.IGNORECASE),
            ),
            ("webhook", re.compile(r"\bwebhook\b", re.IGNORECASE)),
            (
                "queue_consumer",
                re.compile(
                    r"\b(kafka|sqs|consumer|subscriber|celery task|@app\.task)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "cli_entrypoint",
                re.compile(
                    r"\b(argparse|click\.command|typer\.Typer|if __name__ == ['\"]__main__['\"])\b"
                ),
            ),
            (
                "template_render",
                re.compile(
                    r"\b(render_template|string|Response\.render|TemplateResponse|res\.render)\b"
                ),
            ),
            (
                "file_upload",
                re.compile(
                    r"\b(request\.files|UploadFile|multer|multipart/form-data|IFormFile)\b"
                ),
            ),
        ]
        records: List[Dict[str, Any]] = []
        for path in self._code_files():
            text = self._read_text(path)
            if not text:
                continue
            lines = text.splitlines()
            for idx, line in enumerate(lines, start=1):
                for kind, pattern in patterns:
                    if not pattern.search(line):
                        continue
                    symbol = self._nearest_symbol(lines, idx - 1)
                    snippet = line.strip()[:220]
                    records.append(
                        {
                            "kind": kind,
                            "file": self._relative(path),
                            "line": idx,
                            "symbol": symbol,
                            "snippet": snippet,
                        }
                    )
                    if limit is not None and len(records) >= limit:
                        return records
                    break
        return records

    def _nearest_symbol(self, lines: List[str], index: int) -> str:
        patterns = [
            re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            re.compile(r"^\s*async\s+def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\("),
        ]
        for cursor in range(index, min(len(lines), index + 4)):
            raw = lines[cursor]
            for pattern in patterns:
                match = pattern.search(raw)
                if match:
                    return match.group(1)
        return ""

    @function_tool(
        name="audit_inventory",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_inventory(self) -> Dict[str, Any]:
        """
        Inventory repository structure, manifests, frameworks, and likely security-relevant files.
        """
        files = self._iter_repo_files()
        languages: Counter[str] = Counter()
        manifests: List[Dict[str, Any]] = []
        security_relevant: List[str] = []
        for path in files:
            rel = self._relative(path)
            suffix = path.suffix.lower()
            for language, extensions in LANGUAGE_EXTENSIONS.items():
                if suffix in extensions or (
                    path.name == "Dockerfile" and language == "docker"
                ):
                    languages[language] += 1
            if path.name in MANIFEST_FILES:
                manifests.append({"path": rel, "kind": MANIFEST_FILES[path.name]})
            if path.name in SECURITY_RELEVANT_NAMES or any(
                token in rel.lower()
                for token in (
                    "auth",
                    "secret",
                    "config",
                    "docker",
                    ".github/workflows",
                    "k8s",
                    "helm",
                )
            ):
                security_relevant.append(rel)
        entrypoint_candidates = self._scan_entrypoints(limit=min(self.max_matches, 40))
        payload = self._store(
            "audit_inventory",
            {
                "status": "success",
                "stdout": (
                    f"Inventory complete: {len(files)} file(s), {len(manifests)} manifest(s), "
                    f"{len(entrypoint_candidates)} entrypoint candidate(s)."
                ),
                "data": {
                    "languages": dict(languages.most_common()),
                    "manifests": manifests,
                    "framework_hints": self._framework_hints(
                        [
                            path
                            for path in files
                            if path.name in MANIFEST_FILES or path.name == "Dockerfile"
                        ]
                    ),
                    "entrypoint_candidates": entrypoint_candidates,
                    "security_relevant_files": sorted(security_relevant)[
                        : max(20, self.max_matches)
                    ],
                },
            },
        )
        return payload

    @function_tool(
        name="audit_entrypoints",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_entrypoints(self) -> Dict[str, Any]:
        """
        Find repository entrypoints where external input or execution can enter the application.
        """
        records = self._scan_entrypoints(limit=self.max_matches)
        by_kind: Dict[str, int] = Counter(item["kind"] for item in records)
        by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in records:
            by_file[item["file"]].append(item)
        payload = self._store(
            "audit_entrypoints",
            {
                "status": "success",
                "stdout": f"Entrypoint scan found {len(records)} candidate(s) across {len(by_file)} file(s).",
                "data": {
                    "entrypoints": records,
                    "count": len(records),
                    "by_kind": dict(by_kind),
                    "by_file": dict(by_file),
                },
            },
        )
        return payload

    @function_tool(
        name="audit_sink_scan",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_sink_scan(self, category: str = "all") -> Dict[str, Any]:
        """
        Scan for dangerous sink patterns related to injection, SSRF, deserialization, auth, and unsafe writes.

        :param category: Sink family to scan. Use one of `command_exec`, `sql_injection`, `path_traversal`, `ssrf`, `xss_template`, `deserialization`, `file_write`, `crypto_auth`, `redirect_header`, or `all`.
        """
        pattern_map: Dict[str, List[Dict[str, Any]]] = {
            "command_exec": [
                {
                    "regex": re.compile(
                        r"\b(os\.system|subprocess\.(run|Popen|call)|child_process\.(exec|spawn)|Runtime\.getRuntime\(\)\.exec)\b"
                    ),
                    "severity": "high",
                    "confidence": 0.72,
                    "title": "Command execution sink",
                    "rationale": "Command execution primitives often become RCE when user input reaches arguments.",
                    "recommendation": "Trace whether user-controlled input can reach this sink and prefer safe APIs without shell interpretation.",
                },
                {
                    "regex": re.compile(r"\bshell\s*=\s*True\b"),
                    "severity": "high",
                    "confidence": 0.82,
                    "title": "Shell interpretation enabled",
                    "rationale": "Enabling shell interpretation increases command injection risk.",
                    "recommendation": "Avoid `shell=True` and pass command arguments as a list.",
                },
            ],
            "sql_injection": [
                {
                    "regex": re.compile(
                        r"\b(execute|executemany|raw|query)\s*\(\s*f[\"']"
                    ),
                    "severity": "high",
                    "confidence": 0.78,
                    "title": "Interpolated SQL execution",
                    "rationale": "Building SQL queries with string interpolation is a common SQL injection path.",
                    "recommendation": "Use parameterized queries or ORM-safe query builders.",
                },
                {
                    "regex": re.compile(
                        r"\b(cursor|db|session)\.(execute|query)\s*\([^)]*\+"
                    ),
                    "severity": "high",
                    "confidence": 0.74,
                    "title": "Concatenated SQL query",
                    "rationale": "String concatenation into SQL often indicates unescaped user input in queries.",
                    "recommendation": "Replace concatenation with bind parameters.",
                },
            ],
            "path_traversal": [
                {
                    "regex": re.compile(
                        r"\b(open|send_file|send_from_directory|File\(|fs\.(readFile|createReadStream))\b.*\b(request|params|query|input|filename|path)\b"
                    ),
                    "severity": "high",
                    "confidence": 0.68,
                    "title": "User-influenced file path",
                    "rationale": "File APIs fed by request/path input may allow traversal outside intended directories.",
                    "recommendation": "Normalize paths and enforce an allowlisted root before reading or serving files.",
                },
            ],
            "ssrf": [
                {
                    "regex": re.compile(
                        r"\b(requests\.(get|post)|httpx\.(get|post)|fetch|axios\.(get|post))\b.*\b(url|request|params|query|input)\b"
                    ),
                    "severity": "medium",
                    "confidence": 0.63,
                    "title": "Outbound request from variable URL",
                    "rationale": "Dynamic URLs can become SSRF if influenced by external input.",
                    "recommendation": "Validate destination hosts and block internal address ranges.",
                },
            ],
            "xss_template": [
                {
                    "regex": re.compile(
                        r"\b(dangerouslySetInnerHTML|innerHTML\s*=|render_template_string|Markup\(|v-html)\b"
                    ),
                    "severity": "medium",
                    "confidence": 0.69,
                    "title": "Raw HTML rendering sink",
                    "rationale": "Raw HTML rendering bypasses standard escaping and can enable XSS.",
                    "recommendation": "Prefer escaped rendering and sanitize any HTML content that must be rendered.",
                },
            ],
            "deserialization": [
                {
                    "regex": re.compile(
                        r"\b(pickle\.loads|yaml\.load\(|marshal\.loads|jsonpickle\.decode|ObjectInputStream|unserialize\()\b"
                    ),
                    "severity": "high",
                    "confidence": 0.81,
                    "title": "Unsafe deserialization sink",
                    "rationale": "Unsafe deserialization primitives can execute attacker-controlled payloads.",
                    "recommendation": "Use safe loaders and treat serialized input as untrusted.",
                },
            ],
            "file_write": [
                {
                    "regex": re.compile(
                        r"\b(open|write_text|write_bytes|fs\.(writeFile|appendFile)|shutil\.(copy|move))\b.*\b(request|params|query|filename|path)\b"
                    ),
                    "severity": "medium",
                    "confidence": 0.61,
                    "title": "User-influenced file write",
                    "rationale": "User-controlled paths or filenames in write operations may lead to overwrite or traversal issues.",
                    "recommendation": "Constrain writes to a fixed directory and sanitize file names.",
                },
            ],
            "crypto_auth": [
                {
                    "regex": re.compile(
                        r"\b(hashlib\.(md5|sha1)|md5\(|sha1\(|jwt\.decode\([^)]*verify\s*=\s*False|verify\s*=\s*False|ssl\._create_unverified_context)\b"
                    ),
                    "severity": "medium",
                    "confidence": 0.79,
                    "title": "Weak crypto or verification disabled",
                    "rationale": "Weak hashes and disabled verification undermine authentication and transport guarantees.",
                    "recommendation": "Use modern password hashing/signature verification and keep certificate checks enabled.",
                },
            ],
            "redirect_header": [
                {
                    "regex": re.compile(
                        r"\b(redirect|res\.redirect|Response\.redirect|header\(['\"]Location['\"])\b.*\b(request|params|query|next|url)\b"
                    ),
                    "severity": "medium",
                    "confidence": 0.64,
                    "title": "User-influenced redirect",
                    "rationale": "Redirect targets influenced by user input can create open redirect and phishing issues.",
                    "recommendation": "Allowlist redirect destinations or use symbolic route names instead of raw URLs.",
                },
            ],
        }
        categories = list(pattern_map.keys()) if category == "all" else [category]
        findings: List[Dict[str, Any]] = []
        for path in self._code_files():
            text = self._read_text(path)
            if not text:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                for item_category in categories:
                    if item_category not in pattern_map:
                        continue
                    for spec in pattern_map[item_category]:
                        if not spec["regex"].search(line):
                            continue
                        findings.append(
                            self._build_finding(
                                title=str(spec["title"]),
                                category=item_category,
                                severity=str(spec["severity"]),
                                confidence=float(spec["confidence"]),
                                file=self._relative(path),
                                line=line_no,
                                evidence=line.strip()[:260],
                                rationale=str(spec["rationale"]),
                                recommendation=str(spec["recommendation"]),
                                tags=[item_category, "sink"],
                            )
                        )
                        break
                    if len(findings) >= self.max_matches:
                        break
                if len(findings) >= self.max_matches:
                    break
            if len(findings) >= self.max_matches:
                break
        payload = self._store(
            f"audit_sink_scan:{category}",
            {
                "status": "success",
                "stdout": self._summarize_findings(findings, f"Sink scan ({category})"),
                "data": {
                    "category": category,
                    "count": len(findings),
                    "findings": findings,
                },
            },
        )
        return payload

    @function_tool(
        name="audit_secret_scan",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_secret_scan(self) -> Dict[str, Any]:
        """
        Scan for hard-coded credentials, tokens, private keys, and suspicious high-entropy secrets.
        """
        patterns = [
            (
                "Private key block",
                re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
                "critical",
                0.99,
                "Remove committed private keys and rotate them immediately.",
            ),
            (
                "AWS access key",
                re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
                "high",
                0.97,
                "Replace committed AWS access keys with environment-backed secrets.",
            ),
            (
                "GitHub token",
                re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
                "high",
                0.96,
                "Rotate the token and store it in a secret manager.",
            ),
            (
                "Slack token",
                re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
                "high",
                0.95,
                "Rotate the token and remove it from source control.",
            ),
            (
                "Generic credential assignment",
                re.compile(
                    r"(?i)\b(api[_-]?key|secret|token|password|passwd|client_secret)\b\s*[:=]\s*[\"'][^\"']{8,}[\"']"
                ),
                "medium",
                0.7,
                "Move secrets out of source files and into environment-backed configuration.",
            ),
            (
                "JWT-like token",
                re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+\b"),
                "medium",
                0.68,
                "Do not hard-code bearer tokens or JWTs in the repository.",
            ),
        ]
        findings: List[Dict[str, Any]] = []
        for path in self._text_files():
            text = self._read_text(path, max_bytes=250_000)
            if not text:
                continue
            lines = text.splitlines()
            for line_no, line in enumerate(lines, start=1):
                lowered = line.lower()
                if any(token in lowered for token in PLACEHOLDER_SECRET_VALUES):
                    continue
                for title, pattern, severity, confidence, recommendation in patterns:
                    if not pattern.search(line):
                        continue
                    findings.append(
                        self._build_finding(
                            title=title,
                            category="hardcoded_secret",
                            severity=severity,
                            confidence=confidence,
                            file=self._relative(path),
                            line=line_no,
                            evidence=line.strip()[:260],
                            rationale="The line looks like an embedded credential or secret-bearing artifact.",
                            recommendation=recommendation,
                            tags=["secret", "credential"],
                        )
                    )
                    break
                else:
                    candidate = self._extract_entropy_secret(line)
                    if candidate is None:
                        continue
                    findings.append(
                        self._build_finding(
                            title="High-entropy secret-like string",
                            category="hardcoded_secret",
                            severity="medium",
                            confidence=0.62,
                            file=self._relative(path),
                            line=line_no,
                            evidence=line.strip()[:260],
                            rationale="The line assigns a long high-entropy token-like value that may be a secret.",
                            recommendation="Review whether this value is a real secret and move it to managed configuration if so.",
                            tags=["secret", "entropy"],
                        )
                    )
                if len(findings) >= self.max_matches:
                    break
            if len(findings) >= self.max_matches:
                break
        payload = self._store(
            "audit_secret_scan",
            {
                "status": "success",
                "stdout": self._summarize_findings(findings, "Secret scan"),
                "data": {"count": len(findings), "findings": findings},
            },
        )
        return payload

    def _extract_entropy_secret(self, line: str) -> Optional[str]:
        match = re.search(
            r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"']([^\"']{16,})[\"']",
            line,
        )
        if not match:
            return None
        value = match.group(2).strip()
        lowered = value.lower()
        if lowered in PLACEHOLDER_SECRET_VALUES:
            return None
        if _shannon_entropy(value) < 3.5:
            return None
        return value

    @function_tool(
        name="audit_config_scan",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_config_scan(self) -> Dict[str, Any]:
        """
        Scan configuration and deployment files for risky security settings.
        """
        patterns = [
            (
                "Debug mode enabled",
                "debug_config",
                re.compile(r"(?i)\b(debug|devMode)\b\s*[:=]\s*(true|1)\b"),
                "medium",
                0.78,
                "Debug mode can leak stack traces, secrets, and internal behavior.",
                "Disable debug mode in production and gate it behind environment checks.",
            ),
            (
                "Wildcard CORS",
                "cors",
                re.compile(
                    r"(?i)(allow_origins|cors_allowed_origins|Access-Control-Allow-Origin).*\*"
                ),
                "medium",
                0.76,
                "Wildcard CORS often exposes authenticated APIs to unintended origins.",
                "Restrict allowed origins to known frontends.",
            ),
            (
                "TLS verification disabled",
                "tls_verification",
                re.compile(
                    r"\b(verify\s*=\s*False|insecureSkipVerify\s*:\s*true|NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['\"]?0)\b"
                ),
                "high",
                0.86,
                "Disabling TLS verification enables machine-in-the-middle attacks.",
                "Keep certificate validation enabled and trust explicit CA bundles when required.",
            ),
            (
                "Insecure cookie flags",
                "cookie_security",
                re.compile(
                    r"(?i)\b(SESSION_COOKIE_SECURE|COOKIE_SECURE|secure)\b\s*[:=]\s*(false|0)\b"
                ),
                "medium",
                0.69,
                "Session cookies without secure flags are exposed over plaintext channels.",
                "Mark session cookies as Secure and HttpOnly in production.",
            ),
            (
                "Placeholder secret key",
                "default_secret",
                re.compile(
                    r"(?i)\b(secret[_-]?key|jwt[_-]?secret)\b\s*[:=]\s*[\"'](changeme|example|test|dummy|secret)[\"']"
                ),
                "high",
                0.83,
                "Default or placeholder secret keys undermine signing and session protection.",
                "Load a strong secret from managed configuration.",
            ),
            (
                "Privileged container",
                "container_privilege",
                re.compile(
                    r"(?i)\b(privileged|allowPrivilegeEscalation|hostNetwork)\b\s*:\s*true"
                ),
                "high",
                0.8,
                "Privileged containers increase blast radius during compromise.",
                "Drop privilege escalation and only add specific capabilities that are required.",
            ),
            (
                "Container running as root",
                "container_user",
                re.compile(r"^\s*USER\s+root\b", re.IGNORECASE),
                "medium",
                0.7,
                "Running containers as root increases post-compromise impact.",
                "Use a non-root user in container images.",
            ),
        ]
        findings: List[Dict[str, Any]] = []
        for path in self._text_files():
            text = self._read_text(path, max_bytes=300_000)
            if not text:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                for (
                    title,
                    category,
                    pattern,
                    severity,
                    confidence,
                    rationale,
                    recommendation,
                ) in patterns:
                    if not pattern.search(line):
                        continue
                    findings.append(
                        self._build_finding(
                            title=title,
                            category=category,
                            severity=severity,
                            confidence=confidence,
                            file=self._relative(path),
                            line=line_no,
                            evidence=line.strip()[:260],
                            rationale=rationale,
                            recommendation=recommendation,
                            tags=["config", category],
                        )
                    )
                    break
                if len(findings) >= self.max_matches:
                    break
            if len(findings) >= self.max_matches:
                break
        payload = self._store(
            "audit_config_scan",
            {
                "status": "success",
                "stdout": self._summarize_findings(findings, "Config scan"),
                "data": {"count": len(findings), "findings": findings},
            },
        )
        return payload

    @function_tool(
        name="audit_dependency_inventory",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_dependency_inventory(self) -> Dict[str, Any]:
        """
        Inventory dependency manifests, package managers, direct dependencies, and supply-chain risk clues.
        """
        inventory: List[Dict[str, Any]] = []
        for path in self._iter_repo_files():
            if path.name not in MANIFEST_FILES:
                continue
            rel = self._relative(path)
            inventory.append(self._parse_dependency_manifest(path, rel))
        payload = self._store(
            "audit_dependency_inventory",
            {
                "status": "success",
                "stdout": f"Dependency inventory collected from {len(inventory)} manifest(s).",
                "data": {"manifests": inventory, "count": len(inventory)},
            },
        )
        return payload

    def _parse_dependency_manifest(self, path: Path, rel: str) -> Dict[str, Any]:
        kind = MANIFEST_FILES.get(path.name, "unknown")
        text = self._read_text(path, max_bytes=350_000) or ""
        payload: Dict[str, Any] = {
            "path": rel,
            "manager": kind,
            "direct_dependencies": [],
            "locked_dependency_count": 0,
            "git_or_path_dependencies": [],
            "suspicious_dependencies": [],
            "outdated_clues": [],
        }
        if path.name == "requirements.txt":
            deps: List[str] = []
            outdated: List[str] = []
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                base = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
                deps.append(base)
                if "==" not in line:
                    outdated.append(line)
                if "git+" in line or "@" in line:
                    payload["git_or_path_dependencies"].append(line)
            payload["direct_dependencies"] = deps
            payload["outdated_clues"] = outdated
        elif path.name == "package.json":
            try:
                data = json.loads(text or "{}")
            except json.JSONDecodeError:
                data = {}
            deps = dict(data.get("dependencies") or {})
            dev_deps = dict(data.get("devDependencies") or {})
            payload["direct_dependencies"] = sorted(deps.keys())
            payload["locked_dependency_count"] = len(deps) + len(dev_deps)
            payload["git_or_path_dependencies"] = [
                f"{name}:{version}"
                for name, version in {**deps, **dev_deps}.items()
                if str(version).startswith(("file:", "git+", "github:", "workspace:"))
            ]
            payload["outdated_clues"] = [
                f"{name}:{version}"
                for name, version in {**deps, **dev_deps}.items()
                if str(version) in {"*", "latest"}
                or str(version).startswith((">", "^0", "~0"))
            ]
        elif path.name == "pyproject.toml":
            parsed = self._parse_toml(text)
            direct: List[str] = []
            git_or_path: List[str] = []
            outdated: List[str] = []
            project_deps = list(parsed.get("project", {}).get("dependencies") or [])
            for item in project_deps:
                dep = str(item)
                direct.append(re.split(r"[<>=!~\[]", dep, maxsplit=1)[0].strip())
                if "==" not in dep:
                    outdated.append(dep)
            poetry_deps = dict(
                parsed.get("tool", {}).get("poetry", {}).get("dependencies") or {}
            )
            for name, spec in poetry_deps.items():
                if name == "python":
                    continue
                direct.append(str(name))
                if isinstance(spec, dict) and any(
                    key in spec for key in ("git", "path", "url")
                ):
                    git_or_path.append(name)
                elif isinstance(spec, str) and not spec.startswith("=="):
                    outdated.append(f"{name}:{spec}")
            payload["direct_dependencies"] = sorted(set(filter(None, direct)))
            payload["git_or_path_dependencies"] = sorted(set(git_or_path))
            payload["outdated_clues"] = sorted(set(outdated))
        elif path.name in {"Cargo.toml", "Cargo.lock", "go.mod"}:
            payload["direct_dependencies"] = self._parse_simple_module_names(
                text, path.name
            )
            payload["locked_dependency_count"] = len(payload["direct_dependencies"])
        suspicious = [
            dep
            for dep in payload["direct_dependencies"]
            if dep
            and any(ch.isupper() for ch in dep[:1]) is False
            and dep.count("-") >= 3
        ]
        payload["suspicious_dependencies"] = suspicious[:20]
        payload["direct_dependency_count"] = len(payload["direct_dependencies"])
        return payload

    def _parse_toml(self, text: str) -> Dict[str, Any]:
        if tomllib is None:
            return {}
        try:
            return tomllib.loads(text)
        except Exception:
            return {}

    def _parse_simple_module_names(self, text: str, manifest_name: str) -> List[str]:
        names: List[str] = []
        if manifest_name == "Cargo.toml":
            for line in text.splitlines():
                match = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=\s*['\"{]", line)
                if match and match.group(1) != "version":
                    names.append(match.group(1))
        elif manifest_name == "Cargo.lock":
            for line in text.splitlines():
                match = re.match(r'^\s*name\s*=\s*"([^"]+)"', line)
                if match:
                    names.append(match.group(1))
        elif manifest_name == "go.mod":
            for line in text.splitlines():
                match = re.match(r"^\s*require\s+([^\s]+)\s+", line)
                if match:
                    names.append(match.group(1))
        return sorted(set(names))

    @function_tool(
        name="audit_dependency_audit",
        permissions=ToolPermission(filesystem_read=True, command=True, network=True),
        read_only=True,
        needs_approval=True,
    )
    def audit_dependency_audit(self) -> Dict[str, Any]:
        """
        Run optional external dependency auditors such as pip-audit, npm audit, or osv-scanner when available.
        """
        commands = self._dependency_audit_commands()
        if not commands:
            return {
                "status": "error",
                "error_category": "unavailable",
                "stdout": "No supported dependency audit command is available for this repository or environment.",
                "data": {"findings": [], "scans": []},
            }

        findings: List[Dict[str, Any]] = []
        scans: List[Dict[str, Any]] = []
        for scan in commands:
            result = self._run_command(scan["cmd"], cwd=scan.get("cwd"))
            parsed = self._parse_external_audit(scan["kind"], result)
            scans.append(
                {
                    "kind": scan["kind"],
                    "status": result["status"],
                    "command": scan["cmd"],
                    "stderr": result["stderr"],
                }
            )
            findings.extend(parsed)
        payload = self._store(
            "audit_dependency_audit",
            {
                "status": "success",
                "stdout": self._summarize_findings(findings, "Dependency audit"),
                "data": {"count": len(findings), "findings": findings, "scans": scans},
            },
        )
        return payload

    def _dependency_audit_commands(self) -> List[Dict[str, Any]]:
        root = self._repo_root()
        commands: List[Dict[str, Any]] = []
        requirements = root / "requirements.txt"
        package_json = root / "package.json"
        pyproject = root / "pyproject.toml"
        if requirements.exists() and shutil.which("pip-audit"):
            commands.append(
                {
                    "kind": "pip-audit",
                    "cmd": ["pip-audit", "-r", str(requirements), "-f", "json"],
                }
            )
        elif pyproject.exists() and shutil.which("pip-audit"):
            commands.append(
                {
                    "kind": "pip-audit",
                    "cmd": ["pip-audit", "-f", "json"],
                    "cwd": str(root),
                }
            )
        if package_json.exists() and shutil.which("npm"):
            commands.append(
                {
                    "kind": "npm-audit",
                    "cmd": ["npm", "audit", "--json"],
                    "cwd": str(root),
                }
            )
        if shutil.which("osv-scanner"):
            commands.append(
                {
                    "kind": "osv-scanner",
                    "cmd": ["osv-scanner", "--format", "json", "-r", str(root)],
                }
            )
        return commands

    def _run_command(self, cmd: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.external_timeout,
                check=False,
            )
            return {
                "status": "success" if result.returncode in {0, 1} else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "error_category": "unavailable",
                "stdout": "",
                "stderr": f"{cmd[0]} not found",
                "returncode": -1,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "stdout": "",
                "stderr": "command timed out",
                "returncode": -1,
            }

    def _parse_external_audit(
        self, kind: str, result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not result.get("stdout"):
            return []
        try:
            if kind == "pip-audit":
                return self._parse_pip_audit(result["stdout"])
            if kind == "npm-audit":
                return self._parse_npm_audit(result["stdout"])
            if kind == "osv-scanner":
                return self._parse_osv_scanner(result["stdout"])
        except Exception:
            return []
        return []

    def _parse_pip_audit(self, stdout: str) -> List[Dict[str, Any]]:
        data = json.loads(stdout)
        findings: List[Dict[str, Any]] = []
        for dep in list(data.get("dependencies") or []):
            name = str(dep.get("name", "dependency"))
            version = str(dep.get("version", "unknown"))
            for vuln in list(dep.get("vulns") or []):
                vuln_id = str(vuln.get("id", "unknown"))
                findings.append(
                    self._build_finding(
                        title=f"Vulnerable dependency: {name}",
                        category="vulnerable_dependency",
                        severity="high",
                        confidence=0.95,
                        file="requirements.txt",
                        line=1,
                        evidence=f"{name} {version} affected by {vuln_id}",
                        rationale="External dependency audit reported a known vulnerability affecting this package version.",
                        recommendation="Upgrade to a patched version or remove the dependency.",
                        tags=["dependency", "external-audit", vuln_id],
                    )
                )
        return findings

    def _parse_npm_audit(self, stdout: str) -> List[Dict[str, Any]]:
        data = json.loads(stdout)
        vulnerabilities = data.get("vulnerabilities") or {}
        findings: List[Dict[str, Any]] = []
        for name, vuln in vulnerabilities.items():
            severity = str(vuln.get("severity", "medium"))
            via = vuln.get("via") or []
            advisory = via[0] if isinstance(via, list) and via else {}
            title = (
                advisory.get("title")
                if isinstance(advisory, dict)
                else f"npm audit finding for {name}"
            )
            findings.append(
                self._build_finding(
                    title=str(title or f"Vulnerable dependency: {name}"),
                    category="vulnerable_dependency",
                    severity=severity,
                    confidence=0.95,
                    file="package.json",
                    line=1,
                    evidence=f"{name}: {severity}",
                    rationale="npm audit reported a vulnerable package in the dependency graph.",
                    recommendation="Upgrade or replace the vulnerable package and re-run the audit.",
                    tags=["dependency", "external-audit", "npm"],
                )
            )
        return findings

    def _parse_osv_scanner(self, stdout: str) -> List[Dict[str, Any]]:
        data = json.loads(stdout)
        findings: List[Dict[str, Any]] = []
        for result in list(data.get("results") or []):
            source = str(result.get("source", {}).get("path", "dependency-manifest"))
            for pkg in list(result.get("packages") or []):
                package_name = str(pkg.get("package", {}).get("name", "dependency"))
                for vuln in list(pkg.get("vulnerabilities") or []):
                    vuln_id = str(vuln.get("id", "unknown"))
                    findings.append(
                        self._build_finding(
                            title=f"Vulnerable dependency: {package_name}",
                            category="vulnerable_dependency",
                            severity="high",
                            confidence=0.94,
                            file=source,
                            line=1,
                            evidence=f"{package_name} affected by {vuln_id}",
                            rationale="osv-scanner reported a known vulnerability in this dependency.",
                            recommendation="Upgrade to a fixed version or isolate/remove the package.",
                            tags=["dependency", "external-audit", vuln_id],
                        )
                    )
        return findings

    @function_tool(
        name="audit_notes_scan",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_notes_scan(self) -> Dict[str, Any]:
        """
        Find TODO, FIXME, HACK, nosec, and suppressive annotations that may hide security debt.
        """
        pattern = re.compile(
            r"\b(TODO|FIXME|HACK|XXX|SECURITY|nosec|suppress|ignore)\b", re.IGNORECASE
        )
        findings: List[Dict[str, Any]] = []
        for path in self._text_files():
            text = self._read_text(path, max_bytes=250_000)
            if not text:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if not pattern.search(line):
                    continue
                findings.append(
                    self._build_finding(
                        title="Developer security note or suppression",
                        category="audit_note",
                        severity="low",
                        confidence=0.51,
                        file=self._relative(path),
                        line=line_no,
                        evidence=line.strip()[:260],
                        rationale="Developer notes and suppression markers often point to unfinished hardening or intentionally bypassed checks.",
                        recommendation="Review whether this note marks an unresolved vulnerability or intentionally suppressed security control.",
                        tags=["note", "review"],
                    )
                )
                if len(findings) >= self.max_matches:
                    break
            if len(findings) >= self.max_matches:
                break
        payload = self._store(
            "audit_notes_scan",
            {
                "status": "success",
                "stdout": self._summarize_findings(findings, "Audit notes"),
                "data": {"count": len(findings), "findings": findings},
            },
        )
        return payload

    @function_tool(
        name="audit_hotspots",
        permissions=ToolPermission(filesystem_read=True),
        read_only=True,
        concurrency_safe=True,
        needs_approval=True,
    )
    def audit_hotspots(
        self, findings: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate cached audit findings into a ranked shortlist of files and components that deserve manual review.

        :param findings: Optional explicit finding list. If omitted, the tool uses cached findings from earlier audit tools in this toolset instance.
        """
        source_findings = list(findings or self._collect_cached_findings())
        file_scores: Dict[str, float] = defaultdict(float)
        categories: Dict[str, set[str]] = defaultdict(set)
        by_severity = Counter()
        severity_score = {
            "critical": 5.0,
            "high": 3.0,
            "medium": 1.5,
            "low": 0.5,
            "info": 0.25,
        }
        for item in source_findings:
            file_key = str(item.get("file", "unknown"))
            severity = str(item.get("severity", "low"))
            confidence = float(item.get("confidence", 0.5))
            file_scores[file_key] += severity_score.get(severity, 0.5) * confidence
            categories[file_key].add(str(item.get("category", "unknown")))
            by_severity[severity] += 1
        entrypoints = (
            self._session_cache.get("audit_entrypoints", {})
            .get("data", {})
            .get("entrypoints", [])
        )
        for item in entrypoints:
            file_scores[str(item.get("file", "unknown"))] += 0.4
            categories[str(item.get("file", "unknown"))].add(
                f"entrypoint:{item.get('kind', 'unknown')}"
            )
        hotspots = [
            {
                "file": file,
                "score": round(score, 2),
                "categories": sorted(categories[file]),
            }
            for file, score in sorted(
                file_scores.items(), key=lambda pair: pair[1], reverse=True
            )
        ]
        payload = self._store(
            "audit_hotspots",
            {
                "status": "success",
                "stdout": f"Hotspot analysis ranked {len(hotspots)} file(s) for manual review.",
                "data": {
                    "hotspots": hotspots[: min(len(hotspots), 25)],
                    "count": len(hotspots),
                    "summary": {
                        "finding_count": len(source_findings),
                        "severity_counts": dict(by_severity),
                        "entrypoint_count": len(entrypoints),
                    },
                },
            },
        )
        return payload

    def _collect_cached_findings(self) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for payload in self._session_cache.values():
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            if isinstance(data, dict) and isinstance(data.get("findings"), list):
                findings.extend(
                    item for item in data["findings"] if isinstance(item, dict)
                )
        return findings


def security_audit_tools(
    workspace_root: str,
    *,
    include_external: bool = False,
    external_timeout: int = 120,
    max_matches: int = 200,
):
    """Build a registry containing the codebase security audit toolset."""

    from qitos.core.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_toolset(
        SecurityAuditToolSet(
            workspace_root=workspace_root,
            include_external=include_external,
            external_timeout=external_timeout,
            max_matches=max_matches,
        ),
        namespace="",
    )
    return registry


__all__ = ["SecurityAuditToolSet", "security_audit_tools"]
