"""Test that security research tools have needs_approval=True after migration."""

from qitos.core.tool import FunctionTool
from qitos.core.tool_result import ToolResult
from qitos.kit.tool.experimental.security_research.security_audit import SecurityAuditToolSet
from qitos.kit.tool.experimental.security_research.recon_toolset import ReconToolSet
from qitos.kit.tool.experimental.security_research.password_toolset import PasswordToolSet
from qitos.kit.tool.experimental.security_research.web_test_toolset import WebTestToolSet
from qitos.kit.tool.experimental.security_research.network_toolset import NetworkToolSet
from qitos.kit.tool.experimental.security_research.vuln_scan_toolset import VulnScanToolSet
from qitos.kit.tool.experimental.security_research.exploit_toolset import ExploitToolSet


def test_exposed_security_tools_return_typed_errors(tmp_path):
    missing = str(tmp_path / "missing")
    results = [
        ExploitToolSet(workspace_root=str(tmp_path)).port_forward(),
        NetworkToolSet(workspace_root=str(tmp_path)).traffic_analyze(
            pcap_file=missing
        ),
        PasswordToolSet(workspace_root=str(tmp_path)).john_crack(
            hash_file=missing
        ),
        ReconToolSet(
            authorized_targets=["example.com"], workspace_root=str(tmp_path)
        ).host_discovery(target="other.example"),
        VulnScanToolSet(
            authorized_targets=["example.com"], workspace_root=str(tmp_path)
        ).nuclei_scan(target="other.example"),
        WebTestToolSet(
            authorized_targets=["example.com"], workspace_root=str(tmp_path)
        ).sqlmap_scan(target_url="https://other.example"),
    ]

    for result in results:
        assert isinstance(result, ToolResult)
        assert result.status == "error"
        assert result.error
        assert result.output["status"] == "error"


def _all_tools_needs_approval(toolset_instance, tool_names):
    """Assert all named tools on a toolset instance are FunctionTool with needs_approval=True."""
    for name in tool_names:
        tool = getattr(toolset_instance, name)
        assert isinstance(tool, FunctionTool), f"{name}: expected FunctionTool, got {type(tool)}"
        assert tool.meta.needs_approval is True, f"{name}: needs_approval should be True"


class TestSecurityAuditMigration:
    def setup_method(self):
        self.ts = SecurityAuditToolSet(workspace_root=".", include_external=True)

    def test_all_audit_tools_are_function_tool(self):
        for name in [
            "audit_inventory", "audit_entrypoints", "audit_sink_scan",
            "audit_secret_scan", "audit_config_scan", "audit_dependency_inventory",
            "audit_notes_scan", "audit_hotspots", "audit_dependency_audit",
        ]:
            tool = getattr(self.ts, name)
            assert isinstance(tool, FunctionTool), f"{name}: expected FunctionTool, got {type(tool)}"

    def test_all_audit_tools_need_approval(self):
        _all_tools_needs_approval(self.ts, [
            "audit_inventory", "audit_entrypoints", "audit_sink_scan",
            "audit_secret_scan", "audit_config_scan", "audit_dependency_inventory",
            "audit_notes_scan", "audit_hotspots", "audit_dependency_audit",
        ])


class TestReconToolMigration:
    def setup_method(self):
        self.ts = ReconToolSet(workspace_root=".")

    def test_all_recon_tools_need_approval(self):
        _all_tools_needs_approval(self.ts, [
            "host_discovery", "port_scan", "service_scan", "os_detect",
            "subnet_scan", "dns_lookup", "dns_enum", "whois_lookup", "subdomain_enum",
        ])


class TestPasswordToolMigration:
    def setup_method(self):
        self.ts = PasswordToolSet(workspace_root=".")

    def test_all_password_tools_need_approval(self):
        _all_tools_needs_approval(self.ts, [
            "hash_identify", "john_crack", "hashcat_crack",
            "hydra_bruteforce", "wordlist_manage",
        ])


class TestWebTestToolMigration:
    def setup_method(self):
        self.ts = WebTestToolSet(workspace_root=".")

    def test_all_web_test_tools_need_approval(self):
        _all_tools_needs_approval(self.ts, [
            "sqlmap_scan", "dir_bruteforce", "header_check",
            "ssl_check", "tech_detect", "vhost_enum",
        ])


class TestNetworkToolMigration:
    def setup_method(self):
        self.ts = NetworkToolSet(workspace_root=".")

    def test_all_network_tools_need_approval(self):
        _all_tools_needs_approval(self.ts, [
            "packet_capture", "traffic_analyze", "dns_sniff",
            "arp_scan", "traceroute", "http_request", "scapy_craft",
        ])


class TestVulnScanToolMigration:
    def setup_method(self):
        self.ts = VulnScanToolSet(workspace_root=".")

    def test_all_vuln_scan_tools_need_approval(self):
        _all_tools_needs_approval(self.ts, [
            "nuclei_scan", "nikto_scan", "searchsploit",
            "vuln_quick", "cve_query",
        ])


class TestExploitToolMigration:
    def setup_method(self):
        self.ts = ExploitToolSet(workspace_root=".")

    def test_all_exploit_tools_need_approval(self):
        _all_tools_needs_approval(self.ts, [
            "msf_check", "payload_gen", "reverse_shell",
            "port_forward", "priv_esc_check",
        ])
