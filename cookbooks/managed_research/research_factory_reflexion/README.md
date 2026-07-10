# Research Factory: build a Reflexion instance from scratch

This cookbook stands up one Research Factory with one job: create a new
Reflexion implementation in project git, then improve that same instance
continuously. The private Craftax Reflexion cookbook is the reference design,
not the runtime implementation.

```text
Railway: Synth API + Factory scheduler/reactor
    |
    +-- Daytona: disposable B0/C1/C2 workers and paired evaluations
    |       |
    |       +-- project git-server: reflexion_instance/ + experiments/
    |       +-- Synth Wiki: findings and prior-finding carry
    |       +-- Artifact Site: traces, rewards, scorecards, reports
    |
    +-- exe.dev: durable release-audited champion service
```

Railway is the Synth-operated control plane: cookbook users do not run Railway
CLI commands or receive Railway credentials. Synth release engineering proves
logical dev, staging (Railway environment `dev`), and production (Railway
environment `production`) parity. The customer-visible Factory selects Daytona
through its launch profile and creates the exe.dev champion through the typed
Managed Research CloudDeployment API.

## Setup

Install a `synth-ai` version containing the Research Factory, experiment
history, Wiki, CloudDeployment, and WorkProduct APIs, then set `SYNTH_API_KEY`.
Never put raw keys in `factory_plan.json` or evidence files.

Replace `YOUR_PROJECT_ID` in `factory_plan.json`. Preview the exact mutations:

```bash
synth-ai-research-factory-standup \
  --plan @cookbooks/managed_research/research_factory_reflexion/factory_plan.json \
  --dry-run
```

Create the Factory without launching a cycle:

```bash
synth-ai-research-factory-standup \
  --plan @cookbooks/managed_research/research_factory_reflexion/factory_plan.json
```

The first real wake is deliberately separate and operator-confirmed:

```bash
synth-ai-research-factory-standup \
  --plan @cookbooks/managed_research/research_factory_reflexion/factory_plan.json \
  --wake-due --wake-due-launch
```

## Inspect without mutating

```bash
python cookbooks/managed_research/research_factory_reflexion/inspect_factory.py \
  --factory-id "$FACTORY_ID" \
  --project-id "$PROJECT_ID" \
  --output cookbooks/managed_research/research_factory_reflexion/evidence/inspection.json
```

The inspection joins typed Factory status, experiment history, comparable-only
comparison, prompts/configs, traces, costs, Wiki/git receipts, health, and the
operating window, plus hosted Artifact Sites and any retained champion
deployments. It does not read databases, Redis, or local artifact folders.

## Deploy the accepted champion

After B0/C1/C2 and the release audit accept one exact instance head, derive the
deployment directly from that evidence and preview the exact typed request:

The audit is its own preregistered experiment and run, not one of B0/C1/C2.
Attach exactly one `craftax_release_audit_v1_64` experiment result for that run:
64 unique seeds, `evidence_grade=release_evidence`, attested or verified truth,
the paired mean delta, and the scorecard artifact. Put the complete
`craftax_factory.release_audit.v1` receipt under
`metadata.release_audit`, attach the matching completed container execution,
and commit the audit bundle/receipt to the same project git branch. The release
and deployment scripts reject experiment-level membership without this exact
run/evaluation/execution/source binding.

```bash
python cookbooks/managed_research/research_factory_reflexion/deploy_champion.py \
  --release-evidence cookbooks/managed_research/research_factory_reflexion/evidence/release.json
```

For integrations that already materialize a small handoff, the script also
accepts the non-secret `champion_candidate.schema.json` shape via `--candidate`.

The backend derives the git remote and branch from the candidate's Managed
Research project. The caller supplies no remote URL or credential. Launching a
retained VM requires an explicit flag, an exact commit confirmation, and the
final observed receipt produced by mandatory `--wait`:

```bash
python cookbooks/managed_research/research_factory_reflexion/deploy_champion.py \
  --release-evidence cookbooks/managed_research/research_factory_reflexion/evidence/release.json \
  --launch \
  --confirm-source-commit "$SOURCE_COMMIT_SHA" \
  --wait \
  --output cookbooks/managed_research/research_factory_reflexion/evidence/champion.json
```

The `research-reflexion-service` topology checks out the exact accepted source
commit from project git, proves it is an ancestor of the evidence commit at the
recorded branch head, starts only `reflexion_instance`, and probes `/health`.
The CloudDeployment is retained by default; retirement remains a separate,
explicit lifecycle action. The output contract is
`champion_deployment_receipt.schema.json`; only a launched, observed receipt is
eligible for the release packet.

## Acceptance ladder

1. **B0:** create `reflexion_instance/` from scratch and commit its source.
2. **C1/C2:** consume the immediately prior experiment and Gardener findings,
   retain the instance/origin IDs, and advance the immediately prior source
   commit.
3. **Staging:** prove recurrence, cleanup, route failover, budget no-op, and a
   before/restart/after identity triplet.
4. **Release:** consume one locked heldout audit, deploy the champion to exe.dev,
   publish the org-visible Artifact Site, and promote the identical control
   plane through Railway.
5. **24/7:** observe 12 accepted cycles in a rolling 30-day owner window.

`deployment_topology.json` defines the provider roles and receipt fields.
`release_gate.py` is the final fail-closed adoption/blog gate. No synthetic or
source-only packet counts as runtime, production, repeatability, or 24/7 proof.

After collecting the owner-route packet described by
`release_evidence.schema.json`, run:

```bash
python cookbooks/managed_research/research_factory_reflexion/release_gate.py \
  --evidence cookbooks/managed_research/research_factory_reflexion/evidence/release.json \
  --output cookbooks/managed_research/research_factory_reflexion/generated/release_receipt.json \
  --blog-output cookbooks/managed_research/research_factory_reflexion/generated/blog.md
```

The blog is written only after the gate verifies the five clean source
authorities, immutable runtime image, one instance lineage, accepted experiment
bundles, applied Wiki truth, project git source/evidence commits, positive
release audit, hosted Artifact Site, Railway dev/staging/prod parity, Daytona
B0/C1/C2 cleanup, healthy exe.dev champion deployment, and the 12-cycle/30-day
operating proof.
