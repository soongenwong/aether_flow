## Day 5: Multi-Process Orchestration & Pipeline Assembly (5 Hours)

Goal: Wire the components together into a single running pipeline.

Tasks:
Create a master runner that spawns:

One Producer Process (ingesting synthetic ticks into shared memory).

Two Worker Compute Processes (calculating signals in parallel).

One Serving / Query Interface (serving the Pandas aggregations).

Implement clean termination handlers (SIGINT / Ctrl+C) so shared memory
is properly unlinked using shm.unlink() and doesn't leak into your OS
RAM.



### Manual: engine.py
### Vibecode: run_pipeline.py



┌────────────────────────────────────────────────────────┐
│               CAN BE VIBECODED (CODEX)                 │
├────────────────────────────────────────────────────────┤
│ • Interactive CLI / Live Terminal Dashboard            │
│ • Periodic stats logger (printing ticks/sec & latency) │
│ • Multi-symbol stream dispatch logic                   │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                  MUST CODE MANUALLY                    │
├────────────────────────────────────────────────────────┤
│ • `MasterEngine` process lifecycle (spawn, join, kill) │
│ • POSIX signal traps (`signal.SIGINT`, `SIGTERM`)      │
│ • Deterministic shared memory unlinking on failure     │
│ • Worker polling logic tracking `write_head` offsets   │
└────────────────────────────────────────────────────────┘