## 1. Persist EnvDispatch channel execution state

- [x] 1.1 Migration: env collaboration_trigger + environment_agent_sandbox table + clone job type
- [x] 1.2 Focused pgx store: bindings, claim/markReady/markFailed/markDeleting, trigger load/save

## 2. Clone sandbox-instance state through sandboxd

- [x] 2.1 CloneSandboxInstance lifecycle (offline runtime + clone job, compensation)
- [x] 2.2 Cube clone as snapshot/create/delete; completion writes destination external id

## 3. Create project-backed message channels and exact rosters

- [x] 3.1 ResolveMessageRoster + CreateEnvDispatchChannel (group channel, exact members, pending bindings, no Beckham)
- [x] 3.2 resetOne wires message channel creation after project; issue reset byte-for-byte preserved

## 4. Provision and wake only the scratch leader

- [x] 4.1 provisionEnvDispatchAgent (claim, precreate runtime, sandbox, session, markReady, compensate)
- [x] 4.2 dispatchScratchChannelMessage: leader-only provision + channel enqueue + initial trigger + channel-first response

## 5. Lazily provision peers on first mention

- [x] 5.1 routeEnvDispatchChannelAgent hook before session creation (ready/pending/failed/provisioning/deleting)
- [x] 5.2 Persist continuation trigger atomically with enqueue; compensate on enqueue failure

## 6. Copy channel history and resume from the source trigger

- [x] 6.1 ValidateBranchMessageSource before reset fan-out (missing/malformed/unauthorized/roster => 400)
- [x] 6.2 CopyEnvDispatchChannel: members, messages, replies/quotes/threads, read state, thread participants, pending bindings w/ clone sources
- [x] 6.3 Remap trigger to copied entities; provision trigger agent via CloneSandboxInstance (clone when source binding ready, else create from policy); enqueue channel run; save remapped trigger
- [x] 6.4 Append request message.content as nondispatching context without changing the trigger agent
- [x] 6.5 Branch Step 5 tests pass (wakes only trigger w/ cloned sandbox; leaves peers pending w/ clone sources; appends content w/o changing trigger agent)

## 7. Channel-first facades and concurrency-safe cleanup

- [x] 7.1 Channel-to-project resolver + GetEnvDispatchChannelDag, DeleteEnvDispatchChannel, ListChannelEnvCheckpoints handlers
- [x] 7.2 Register channel-first routes in cmd/server/router.go
- [x] 7.3 Serialized cleanup: lock env+bindings, mark deleting, compensate in-flight provisioning, delete in FK-safe order, idempotent

## 8. AReaL message client channel-first

- [ ] 8.1 EnvDispatchHandle (channel_id primary for message, project_id for issue)
- [ ] 8.2 create_env_dispatch returns handle; DAG/checkpoint/cleanup route by dispatch_type
- [ ] 8.3 multi_agent_workflow + multica_dag_client pass handle; issue callers stay source-compatible

## 9. Regression verification and protocol documentation

- [ ] 9.1 Document final request/response, channel-first routes, leader-only wake, lazy provisioning, branch errors in multica_environment_protocol.md
- [ ] 9.2 Go suites pass (service/handler/migrations/cmd/multica/cmd/server)
- [ ] 9.3 Python suites pass (test_env_dispatch_client, test_multica_dag_client)
- [ ] 9.4 gofmt, go test ./..., graphify update, pre-commit pass
- [ ] 9.5 Final invariant review (no default-runtime fallback; only intended files staged)
