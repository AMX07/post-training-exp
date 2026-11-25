# Vendored sources

- `nanochat_optim.py` and `reference/chat_rl.py.txt`: Andrej Karpathy,
  karpathy/nanochat, commit 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd,
  MIT license in LICENSE.nanochat. Optimizer modification: replace its
  nanochat.common compute-dtype import with standalone torch dtype selection.
- `drgrpo_grader.py`: copied unchanged from stanford-cs336/assignment5-alignment,
  commit c2734a26308710949fe13226960a1e8cece94b7e. Copyright/license notices are
  retained in the file; Apache 2.0 license included as LICENSE.apache-2.0.
  This grader credits sail-sg/understand-r1-zero and its upstream contributors.
- `r1_zero.prompt`: same CS336 commit, DeepSeek R1-Zero prompt as documented
  in the user's CS336 Spring 2026 Assignment 5 PDF, section 3.2.
