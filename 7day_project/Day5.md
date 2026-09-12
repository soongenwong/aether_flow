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