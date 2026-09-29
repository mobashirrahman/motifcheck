"""G0/G1: freeze the protocol, acquire sources, record provenance."""

from mc_common import CONFIG, GATES, MANIFEST, cli, run


rule protocol_frozen:
    """Freeze the protocol and its amendment log before any modelling."""
    input:
        config=CONFIG,
    output:
        record=GATES / "G0.json",
    log:
        "logs/protocol_frozen.log",
    shell:
        run(cli("protocol-freeze", "--config", input.config,
                "--record", output.record)) + " 2>&1 | tee {log}"


rule validate_sources:
    """Download and verify every asset; write the immutable source manifest.

    Emits G1 and exits non-zero on any failure, so a missing or checksum-mismatched
    source stops the workflow instead of producing a quietly smaller dataset.
    """
    input:
        config=CONFIG,
        protocol=GATES / "G0.json",
    output:
        manifest=MANIFEST,
        record=GATES / "G1.json",
    log:
        "logs/validate_sources.log",
    shell:
        run(cli("validate-sources", "--config", input.config,
                "--manifest", output.manifest,
                "--record", output.record)) + " 2>&1 | tee {log}"
