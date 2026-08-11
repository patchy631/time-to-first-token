# Learn LLM Inference Serving by Shipping One Service

A 10-week, 30-minutes-a-day roadmap for engineers who want to actually run LLM inference in production, not just read about it.

Fifty sessions. Every one of them feeds a single artifact: an OpenAI-compatible inference service that you deploy on a rented GPU, instrument, load test past 1000 concurrent requests, optimize with quantization and speculative decoding, put a cost-aware router in front of, and publish as a reproducible benchmark.

The alternative approach, running seventeen disconnected experiments, spends most of its time on setup. One service that keeps growing gets you the same coverage and leaves you with something to show.

---

## Who this is for

You should be comfortable with Python, transformers at the architecture level, and the command line. You do not need prior serving, Kubernetes, or CUDA experience.

If you already know a topic, the sessions marked **skim** are the ones to compress. Do not compress the build sessions, they are the point.

## What you will have at the end

- A serving stack you configured, instrumented, and tuned yourself
- Grafana dashboards showing TTFT, inter-token latency, throughput, queue depth, and cost per request
- A load test harness that reproducibly drives 1000+ concurrent requests
- Benchmarked variants across FP16, FP8, INT4, speculative decoding, and KV eviction
- A cost/latency/quality router with per-request token budgeting
- A published benchmark writeup with pinned versions and reproducible commands

## Time commitment

| | |
|---|---|
| Session length | 30 minutes |
| Sessions per week | 5, plus 2 buffer days |
| Total | 10 weeks, 50 sessions, 25 hours |
| GPU cost | Roughly one 24GB card rented by the half hour, plus two H100 sessions |

Buffer days exist so that missing a Tuesday does not collapse the plan. They are for catching up, not for new material.

---

## How the roadmap is ordered

The order here is deliberate and differs from how most people write this list down. Five decisions drive it.

**The roofline model comes first.** Every optimization later is a move on the same plot. Quantization attacks memory-bandwidth-bound decode. Continuous batching raises arithmetic intensity toward the compute roof. Speculative decoding spends FLOPs, which are cheap in the memory-bound regime, to cut sequential memory loads. Disaggregation exists because prefill is compute-bound and decode is bandwidth-bound and they fight over one GPU. Without this model, the rest is a bag of tricks.

**Measurement comes before optimization.** Instrumentation lands in week 3 and load testing in week 5, well before the tuning knobs. Nothing after them is verifiable without them, and a public benchmark is mostly a credible measurement harness with a model attached.

**Paged attention and continuous batching are not build exercises.** vLLM implements both, and chunked prefill is the default scheduling strategy in vLLM and SGLang. Reimplementing them teaches less than reading the block manager and the scheduler until you can explain from the code why memory waste drops under 4 percent and why the GPU stops idling between requests.

**The router sits next to unit economics.** A router that picks a backend by cost, latency, and quality is the one component that forces a dollar figure and a latency budget into code. Studying economics the same week turns the reading into a routing policy instead of a blog post you agreed with.

**Edge deployment is optional and last.** ONNX Runtime, TensorRT-LLM, and WebLLM are client-side and embedded runtimes. They share almost no operational surface with the datacenter stack the other nine weeks build.

---

## Setup

### GPU access

A 7-8B model needs roughly one 24GB GPU. Rent by the half hour and shut the instance down between sessions.

| Provider | Good for |
|---|---|
| [RunPod](https://www.runpod.io) | Per-second billing, quick pods |
| [Modal](https://modal.com) | Serverless, best for benchmark sweep scripts |
| [Lambda](https://lambda.ai) / [vast.ai](https://vast.ai) | Cheap on-demand and marketplace GPUs |
| [Colab](https://colab.research.google.com) | Pure-Python sessions, no serving |

All of week 1, most of week 3, the week 6 lecture days, and all of week 9 reading need no GPU at all. Batch your rentals around the build days. One H100 is needed for exactly two sessions: the 1000-concurrent test in week 5, and disaggregation in week 8 if you run it for real.

### Tools you will install

```bash
pip install vllm
pip install guidellm
pip install sglang
```

Plus Docker for the Prometheus and Grafana stack, and `kind` or a small managed cluster for week 8.

---

## The roadmap

Legend: **read** sessions load context, **build** sessions produce output. **skim** marks material you may already know conceptually, where the value is in the hands-on part.

---

### Week 1: The mental model

**Focus.** Install the one model everything else hangs on. Roofline, arithmetic intensity, and why decode waits on memory while prefill waits on compute.

**Medium.** Video-first. Roofline reasoning benefits from live derivation and diagrams.

**Deliverable.** A hand-derived arithmetic intensity figure for the model you will serve, and the ability to say which phase is bandwidth-bound and why.

| Day | Session |
|---|---|
| Mon | *read* · [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He), first half through the compute vs memory-bandwidth vs overhead regimes. |
| Tue | *read* · Finish the post. Operator fusion, overhead, and why adding FLOPS does nothing for a memory-bandwidth-bound kernel. |
| Wed | *read* · [Stanford CS336 Lecture 5: GPUs](https://www.youtube.com/watch?v=6OBtO9niT00) (Percy Liang, Tatsunori Hashimoto), first 30 min: execution model and memory hierarchy. [Slides](https://github.com/stanford-cs336/spring2025-lectures). |
| Thu | *read* · CS336 Lecture 5, second 30 min: arithmetic intensity, roofline, why data movement dominates. Text version: [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (kipply). |
| Fri | *build, skim* · Hand-derive arithmetic intensity for your target model. The value is producing your own figure, not re-reading the explanation. |
| Buffer | [CS336 Lecture 10: Inference](https://www.youtube.com/watch?v=fcgPYo3OtV0), then the prefill vs decode opening of the [Databricks post](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices). |

---

### Week 2: vLLM: deploy it, then read its internals

**Focus.** Stand the service up, then go into the block manager and scheduler until the paging design is obvious from the code.

**Medium.** Text-first for PagedAttention, where the paper is more precise. Video-first for the V1 architecture.

**Deliverable.** An OpenAI-compatible endpoint serving a 7-8B model, plus your own notes explaining PagedAttention and the V1 scheduler from source.

| Day | Session |
|---|---|
| Mon | *read, skim* · [PagedAttention announcement](https://blog.vllm.ai/2023/06/20/vllm.html). Prior systems waste 60-80% of KV memory on fragmentation and over-reservation; fixed-size blocks cut waste under 4%. Then skim the [SOSP paper](https://arxiv.org/abs/2309.06180) block-table section. |
| Tue | *build* · Serve Llama-3.1-8B-Instruct or Qwen2.5-7B behind vLLM's OpenAI-compatible server. Save the launch command; you reuse it for ten weeks. [Modal example](https://modal.com/docs/examples/vllm_inference). |
| Wed | *read* · [vLLM Office Hours 22: Intro to vLLM V1](https://www.youtube.com/watch?v=jmzIvQZCLZM) (Michael Goin, Red Hat). Written version: [vLLM V1 architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html). |
| Thu | *read* · [Inside vLLM: Anatomy of a High-Throughput Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html), scheduler and block manager sections. |
| Fri | *build* · Follow the block table code path in [the repo](https://github.com/vllm-project/vllm). Write three sentences on how a request's blocks get looked up. If you cannot, read it again. |
| Buffer | Re-read the V1 blog's scheduler section against the code, and note where the blog simplifies. |

---

### Week 3: Measurement infrastructure

**Focus.** Build the lens before the optimizations. Everything after this week is only legible because of this week.

**Medium.** Text-first. The writing on benchmarking methodology is stronger than any available video.

**Deliverable.** A live Grafana dashboard showing TTFT, inter-token latency, throughput, and queue depth on your own service.

| Day | Session |
|---|---|
| Mon | *read* · [vLLM metrics design doc](https://docs.vllm.ai/en/latest/design/v1/metrics/). Learn what `num_requests_running`, `num_requests_waiting`, and the latency histograms actually measure. |
| Tue | *build* · Stand up the [Prometheus and Grafana stack](https://docs.vllm.ai/en/latest/examples/online_serving/prometheus_grafana/) against your service. |
| Wed | *build* · Import the dashboard JSON, confirm live panels move under a trickle of load. Fix the scrape config now, not in week 5 under 1000 concurrent requests. |
| Thu | *read* · [Databricks performance engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) and [NVIDIA benchmarking fundamentals](https://developer.nvidia.com/blog/llm-inference-benchmarking-fundamental-concepts/). Map every metric they define to a panel. Panels without a definition get deleted. |
| Fri | *read* · [How to Benchmark LLM Engines](https://modal.com/llm-almanac/how-to-benchmark) (Modal). Why you sweep request rate rather than pick one, and why the saturation point gets discarded. |
| Buffer | Optional: [vLLM Office Hours 21: Production Stack Deep Dive](https://www.youtube.com/watch?v=0ZVu0A4wWQg) for observability in a deployed setting. |

---

### Week 4: SGLang, RadixAttention, and batching internals

**Focus.** The contrast that teaches the design space: fixed-block paging versus tree-structured prefix reuse.

**Medium.** Mixed. The RadixAttention talk is from the project lead. Chunked prefill is text-only.

**Deliverable.** An SGLang variant of the same service, and a first prefix-reuse comparison against vLLM.

| Day | Session |
|---|---|
| Mon | *read* · [RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/). Reports up to 5x higher throughput on prefix-heavy workloads with no noticeable overhead when there are no cache hits, which tells you which workloads it wins on. |
| Tue | *build* · Deploy [SGLang](https://docs.sglang.io) with the same model as week 2. Same hardware, same prompt, so the engine is the only variable. |
| Wed | *read* · [Efficient LLM Inference with SGLang](https://www.youtube.com/watch?v=Ny4xxErgFgQ) (Lianmin Zheng, SGLang lead). Paper: [SGLang, NeurIPS 2024](https://arxiv.org/abs/2312.07104). |
| Thu | *read, skim* · [Continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference) (Anyscale), then [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu). Focus on iteration-level scheduling and the heavy-tailed length assumption that makes the numbers real. |
| Fri | *read* · [Sarathi-Serve](https://arxiv.org/abs/2403.02310) on chunked prefill and stall-free scheduling. This is why chunked prefill became the default in both engines. |
| Buffer | [SGLang v0.4 zero-overhead scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), then run a shared-system-prompt workload against both engines and watch the TTFT gap. |

---

### Week 5: Load testing to 1000 concurrent

**Focus.** Build the reproducible harness and learn what makes a benchmark defensible before you have anything to defend.

**Medium.** Text-first. Documentation and hands-on only.

**Deliverable.** A scripted concurrency sweep past 1000 concurrent requests with p50, p95, and p99 output, cross-checked by two independent tools.

| Day | Session |
|---|---|
| Mon | *read* · [GuideLLM introduction](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference) and install it. It sweeps from synchronous baseline to saturation rather than testing one arbitrary load level. |
| Tue | *build* · Run a [GuideLLM](https://github.com/vllm-project/guidellm) sweep with fixed input and output token lengths. Fixed lengths determine KV cache size and therefore the entire result. |
| Wed | *build* · Run [`vllm bench serve`](https://docs.vllm.ai/en/latest/benchmarking/cli/) on the same workload as a cross-check. Two tools disagreeing is information, not a problem. |
| Thu | *build* · [genai-perf](https://github.com/triton-inference-server/perf_analyzer) concurrency sweep at 1, 2, 4 up to 128. Find where throughput saturates while latency degrades. That knee is the only interesting point on the curve. |
| Fri | *build* · Rent an H100 for this session only. Push past 1000 concurrent, watching KV cache utilization and `num_requests_waiting` for preemption. Read [vLLM tuning strategies](https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning) while it runs. |
| Buffer | Write the pre-publish checklist into your benchmark repo. |

---

### Week 6: Quantization tradeoffs

**Focus.** The first tuning knob, and the first week your harness earns its keep.

**Medium.** Video-first. The lectures are the entry point, the papers are the deep dive.

**Deliverable.** FP8 and INT4/AWQ variants benchmarked against the FP16 baseline on both throughput and a quality proxy.

| Day | Session |
|---|---|
| Mon | *read* · [MIT 6.5940 Lecture 5: Quantization Part I](https://www.youtube.com/watch?v=TSc_BibWRhM) (Song Han), first half. [Course page](https://hanlab.mit.edu/courses/2024-fall-65940). |
| Tue | *read* · Finish Lecture 5, start Lecture 6 for AWQ, GPTQ, and SmoothQuant depth. SmoothQuant and AWQ came out of this lab. |
| Wed | *read, skim* · Quantization section of [Lilian Weng's survey](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/). Write down which method each engine you run actually implements. |
| Thu | *build* · Serve an AWQ or GPTQ checkpoint and an FP8 variant. Capture throughput and memory under the same sweep as week 5. [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/). |
| Fri | *build* · Run a fixed eval set or perplexity proxy across FP16, FP8, and INT4. This table is the deliverable, not the speedup number. |
| Buffer | [QLoRA talk](https://www.youtube.com/watch?v=fQirE9N5q_Y) (Tim Dettmers, author of LLM.int8() and QLoRA), then [LLM Compressor office hours](https://www.youtube.com/watch?v=WVenRmF4dPY). |

---

### Week 7: Speculative decoding and KV eviction

**Focus.** Two knobs where the concept is widely known but the implementation detail is not.

**Medium.** Video-first for speculative decoding, taught by the person who wrote vLLM's implementation. Text-first for KV eviction.

**Deliverable.** A speculative decoding variant and a long-context eviction configuration, both benchmarked, including negative results.

| Day | Session |
|---|---|
| Mon | *read, skim* · [Speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) for the method-selection table, then [the vLLM benchmark post](https://blog.vllm.ai/2024/10/17/spec-decode.html): up to 2.8x at QPS 1, but 1.4-1.8x slowdown at high QPS. The crossover is the finding, not the speedup. |
| Tue | *read* · [GPU MODE Lecture 22: Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4) (Cade Daniel, Anyscale), first half: proposers, scorers, verifiers, draft model worker. |
| Wed | *build* · Enable speculative decoding and benchmark inter-token latency at both low and high QPS. Finding your own crossover point is the session. |
| Thu | *read* · [StreamingLLM](https://arxiv.org/abs/2309.17453) (Guangxuan Xiao et al, ICLR 2024): up to 22.2x speedup over sliding-window recomputation, stable past 4M tokens. Optional video: [StreamingLLM and DuoAttention](https://www.csail.mit.edu/event/scale-ml-guangxuan-xiao-streamingllm-and-duoattention-efficient-and-effective-long-sequence) (MIT CSAIL). |
| Fri | *build* · Configure a long-context workload and an eviction strategy, then measure memory and TTFT. Long context is where eviction stops being theoretical. |
| Buffer | Finish GPU MODE Lecture 22, then skim [speculators](https://github.com/vllm-project/speculators) and [its office hours](https://www.youtube.com/watch?v=2ISAr_JVGLs). |

---

### Week 8: Disaggregated serving and Kubernetes

**Focus.** The prefill and decode asymmetry from week 1 becomes an architecture, then the operations layer that runs it.

**Medium.** Video-first for disaggregation. Text-first for Kubernetes, which changes too fast for video to stay current.

**Deliverable.** Your service on Kubernetes with autoscaling driven by queue depth rather than CPU.

| Day | Session |
|---|---|
| Mon | *read* · [DistServe at OSDI 2024](https://www.youtube.com/watch?v=WwJvecXOeUA) (Yinmin Zhong, presenting author). Reports serving 7.4x more requests or holding 12.6x tighter SLOs. [Paper](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin). |
| Tue | *read* · [Splitwise](https://arxiv.org/abs/2311.18677) (Microsoft, ISCA 2024): H100-for-prefill and power-capped-A100-for-decode gave 1.4x throughput at 20% lower cost. Then [vLLM disaggregated prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/). |
| Wed | *read* · [Mooncake](https://arxiv.org/abs/2407.00079) (FAST 2025 best paper), KV-cache-centric, in production at Kimi. Then [SGLang PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation). Compare how each engine splits the phases. |
| Thu | *build* · Deploy with the [vLLM production stack](https://github.com/vllm-project/production-stack) Helm charts, on a managed cluster or `kind`. |
| Fri | *build* · Scale on the `num_requests_waiting` custom metric, [not CPU](https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda). CPU-based autoscaling on a GPU service is the mistake this session prevents. |
| Buffer | Place the ecosystem on the map: [PyTorch disaggregated inference](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/), [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo), [llm-d](https://llm-d.ai). |

---

### Week 9: Unit economics and the router

**Focus.** Turn cost reading into code. This is where the roadmap stops being about latency and starts being about margin.

**Medium.** Text-first.

**Deliverable.** A router that picks a backend by cost, latency, and quality, plus per-request token budgeting with cost on the dashboard.

| Day | Session |
|---|---|
| Mon | *read* · [LLM Inference Economics from First Principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first). GPU-hour-to-token math, and why utilization rather than price per hour drives cost. |
| Tue | *build* · Take the formula (hourly rate ÷ (peak tokens/sec × 3600 × utilization) × 1M) and re-derive cost per million tokens for your own GPU using your week 5 throughput. [Worked example](https://www.digitalocean.com/community/tutorials/llm-inference-cost). |
| Wed | *read* · [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/). A 33x serving cost reduction via int8 on weights, activations, and KV cache plus multi-query attention and cross-layer KV sharing. Every lever is one you studied in weeks 6 and 7. |
| Thu | *build* · Build the router. A thin service that picks cheap-small or expensive-large under a cost/latency/quality policy, logging the chosen path and its cost to Prometheus. |
| Fri | *build* · Token budgeting middleware that caps and accounts tokens per request. Put per-request cost on the dashboard next to latency. |
| Buffer | [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) as a consolidation pass over weeks 1-9. |

---

### Week 10: Publish, then build the reading habit

**Focus.** Ship the artifact and set up the inputs that keep it current.

**Deliverable.** A published repo and writeup with pinned versions and reproducible commands, plus a weekly research session on the calendar.

| Day | Session |
|---|---|
| Mon | *build* · Draft the writeup: hardware, model, exact token lengths, concurrency sweep, p50/p95/p99 for TTFT and ITL, throughput. Run the checklist below against it first. |
| Tue | *build* · Add quantization, speculative decoding, and KV eviction as labeled variants with reproducible commands and pinned versions. Publish. |
| Wed | *read* · Set up the weekly habit. [arXiv cs.DC](https://arxiv.org/list/cs.DC/recent), [MLSys](https://mlsys.org), OSDI, NeurIPS efficiency tracks. Queue three items. |
| Thu | *read* · First habit session. One paper's abstract, method figure, and evaluation in 30 minutes. Not the whole paper. |
| Fri | *optional* · Edge sampler: [ONNX Runtime](https://onnxruntime.ai), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [WebLLM](https://webllm.mlc.ai). Note these are client-side runtimes, not an extension of what you built. |
| Buffer | Circulate the benchmark for feedback. Criticism you can answer is proof the harness is real. |

---

## Tracking your progress

Three options, in order of how little setup they need.

**1. The web page.** Open `index.html`, or the GitHub Pages version, and tick sessions as you finish them. Progress saves to your browser's local storage, so closing the tab or restarting your machine does not lose it. It is per browser and per device, with no account and no server involved.

To move progress to another device, use **Copy sync link**, which packs your state into a 12-character code in the URL, or **Export**, which downloads a JSON file you can re-import later. Both are also your backup if you clear browser data.

If you open the page in private browsing or inside a sandboxed frame, local storage may be blocked. The page detects this and warns you, and ticks will still work for that visit.

**2. A GitHub issue.** Fork the repo, open a new issue from the progress template, and tick the checkboxes there. GitHub stores task-list state on its own servers, tied to your account, so it works across every device you sign in from and gives you a dated comment history of when you finished each week. This is the best option if you want your progress to outlive a browser profile.

**3. The README itself.** Fork the repo and tick the checklists in your copy. Your commit history becomes the record.

### On using a database

A server-side store, SQLite or otherwise, is the wrong tool here and worth explaining why. It would require hosting, accounts so the server knows whose progress is whose, session handling, a privacy policy for the data you are now holding, and ongoing maintenance for a project whose entire value is a list of links. SQLite in particular is a poor fit for a deployed web app, because most cheap hosting has an ephemeral filesystem that discards the database file on redeploy.

Local storage gives the same outcome for a single reader with none of that. The one thing it does not give is cross-device sync, and the sync link and JSON export cover that case without a backend.

If you genuinely want a hosted version, for example a cohort going through this together with a shared leaderboard, the realistic stack is GitHub OAuth for identity plus a managed Postgres or Turso instance, not a SQLite file on a web server.

## Before you publish a benchmark

These are the mistakes that get public inference benchmarks taken apart. Read them in week 5, apply them in week 10.

- [ ] Sweep the request rate. A single concurrency level is an anecdote.
- [ ] Report exact input and output token lengths. KV cache size and therefore throughput depend entirely on them.
- [ ] Report p50, p95, and p99. Means hide the tail that users feel.
- [ ] Keep TTFT and inter-token latency separate. Collapsing prefill and decode into one number destroys the finding.
- [ ] Do not measure at the saturation point where the server is preempting. Report the knee, not the collapse.
- [ ] Avoid synthetic uniform-length traffic. Uniform lengths hide the heavy tail that makes continuous batching matter.
- [ ] Account for tokenizer differences between engines before comparing tokens per second.
- [ ] Pin engine versions and publish exact commands, so a critic can reproduce rather than speculate.

## Decision points

Moments where following the schedule blindly is the wrong move.

| If | Then |
|---|---|
| **Week 5.** TTFT spikes and `num_requests_waiting` climbs before target concurrency. | The server is preempting. Stop and tune `--max-num-seqs`, `--gpu-memory-utilization`, and chunked prefill first. A run measured at the preemption point is not publishable. |
| **Week 6.** INT4 costs more than 1-2% on your quality proxy for the throughput it returns. | Keep FP16 or FP8 as the default. Treat INT4 as a memory-pressure lever, not a throughput default. |
| **Week 7.** Speculative decoding slows you down at your real request rate. | Expected. Gains appear at low query rate and vanish when the GPU is compute-saturated. Publish the crossover point, it is more useful than the speedup. |
| **Week 8.** Kubernetes starts eating whole sessions on cluster plumbing. | Use a managed cluster, or fall back to single-node Docker plus the KEDA reading. The autoscaling signal matters more than a bespoke cluster. |

---

## Reference

### Papers

| Paper | Venue | Topic |
|---|---|---|
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | SOSP 2023 | KV cache paging |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | OSDI 2022 | Continuous batching |
| [SGLang](https://arxiv.org/abs/2312.07104) | NeurIPS 2024 | RadixAttention, prefix reuse |
| [SARATHI](https://arxiv.org/abs/2308.16369) / [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | OSDI 2024 | Chunked prefill |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | OSDI 2024 | Prefill/decode disaggregation |
| [Splitwise](https://arxiv.org/abs/2311.18677) | ISCA 2024 | Phase splitting across GPU types |
| [Mooncake](https://arxiv.org/abs/2407.00079) | FAST 2025 | KV-cache-centric serving |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | ICLR 2024 | Attention sinks, long context |
| [EAGLE](https://arxiv.org/abs/2401.15077) | arXiv | Speculative sampling |

For GPTQ, AWQ, SmoothQuant, LLM.int8(), Medusa, H2O, SnapKV, and the FlashAttention family, use [Lilian Weng's survey](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) and the [NVIDIA optimization survey](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/), which cite them directly. Search exact titles rather than guessing arXiv IDs.

### Courses and series

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/), taught by Percy Liang and Tatsunori Hashimoto. Lectures 5 (GPUs), 6 (Kernels), 7-8 (Parallelism), 10 (Inference).
- [MIT 6.5940 EfficientML.ai](https://hanlab.mit.edu/courses/2024-fall-65940), taught by Song Han. Fall 2023 and Fall 2024 recordings both work.
- [vLLM Office Hours](https://www.youtube.com/@vllm-project), bi-weekly, from the engine team.
- [GPU MODE](https://www.youtube.com/@GPUMODE), kernel and systems-level lectures.
- [USENIX](https://www.usenix.org/conferences), OSDI, NSDI, and ATC talks by paper authors.

### Tools

| Tool | Use |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Primary serving engine |
| [SGLang](https://docs.sglang.io) | Prefix-reuse comparison |
| [GuideLLM](https://github.com/vllm-project/guidellm) | SLO-aware rate sweeps |
| [genai-perf](https://github.com/triton-inference-server/perf_analyzer) | TTFT and ITL measurement |
| [LLMPerf](https://github.com/ray-project/llmperf) | Hosted endpoint benchmarking |
| [production-stack](https://github.com/vllm-project/production-stack) | Kubernetes deployment |

### Reading habit

One 30-minute session a week: scan titles on arXiv cs.DC and the systems conferences, read exactly one paper's abstract, method figure, and evaluation, and track the applied writeups on the [vLLM](https://blog.vllm.ai) and [LMSYS](https://lmsys.org/blog/) blogs and the [Modal Almanac](https://modal.com/llm-almanac/summary). Read the systems paper, skip the model-release press.

---

## Notes on the sources

Every video here has a named speaker from a university, an engine team, or a conference, and is paired with a text covering the same material, so you can read instead of watch. Where no video met that bar, the week is text-only rather than padded.

A few things to keep in mind:

- Timestamps are targets. Talks get re-uploaded and re-cut, so use the on-video chapter markers to place split points.
- MIT 6.5940 was not offered in Fall 2025. Fall 2023 and Fall 2024 recordings both exist and either works.
- vLLM serves its blog at two URL shapes and its docs change with versions. If a link 404s, search the exact title rather than editing the path.
- Performance figures quoted here are the sources' own claims from their papers and blogs, not independently reproduced. Treat them as the number that source is willing to defend, not as ground truth for your hardware.
- Cost per million tokens and GPU hourly rates move fast. The economics reading is valuable for its method. Re-derive every number against your own hardware.

## Contributing

Corrections and additions are welcome, particularly:

- Broken or moved links
- Newer primary sources that supersede something listed here
- Videos that meet the bar (named speaker, credible institution, paired text) for a week currently marked text-only
- Your own benchmark writeup, if you finish the roadmap

Open an issue or a PR.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it, adapt it, teach from it.
