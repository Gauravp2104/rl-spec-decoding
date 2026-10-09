# [BP-06] Online loop

Tracks #14. Branch: `bp/06-online-loop`.

- Bounded sample pool (`max_sample_pool_size`, `buffer_threshold`), BP-03 trainer, versioned weight publish every N steps.
- Modes: **sync** (serve B requests ↔ train M steps; fixed Δ) and **async** (two threads; real Δ).
- Starts: **day-0** (random draft) and **with-draft** (BP-04 checkpoint).
- **Done when:** a 300-request day-0 run on the mixed stream shows τ̄ rising.
