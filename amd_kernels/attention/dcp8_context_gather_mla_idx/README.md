# DCP8 context gather

## Contract

Eight gfx950 ranks push byte-preserving MLA and indexer context rows into peer workspaces. All ranks must participate in the same collective order. Indexer values and scales retain their packed page layout. Workspace allocation and teardown are collective; callers must keep buffers alive until all GPU work completes.

## Entry points

- `DcpContextGather`: owns the IPC workspace, gathers context and synchronizes peer exchanges.
- `compile_context_push`: builds the FlyDSL gather variant selected by row layout.

## Validation status

Draft implementation. Repository-native multi-rank reference tests and graph replay benchmarks are pending; no standalone performance claim is made here. Host API validation and process-group lifetime review remain open.
