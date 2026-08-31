---
{"dg-publish":true,"permalink":"/jax-mambas/","updated":"2026-08-23T10:12:03.000-05:00","dg-note-properties":{}}
---

# JAX Mambas
[GitLab (Main Repo)](https://gitlab.com/RaameshB/JAX-Mambas) | [GitHub (Mirror)](https://github.com/RaameshB/JAX-Mambas)
## Summary
A project where I am doing JAX+Flax NNX implementation of Mamba-1, 2 and 3. This includes replicating some of their experiments (as my compute budget allows), and porting their CUDA kernels to either JAX/OpenXLA's FFI or (ideally) Pallas. 

My objective is to provide both paper-accurate and PyTorch-repo-accurate implementations. This, in my opinion, is important as I found that the experiments conducted in the Mamba-1 paper were replicated by the paper-accurate version but not the repo accurate version if the repo accurate version was plugged in witht the paper's hyperparams.

## Progress
### Mamba
My work on this is complete. 
- I've replicated the Induction Heads experiment to a generalization seqeunce length of 32,768 (the longest I could go to due to compute/memory constraints)
- I have manually (very spooky in the age) implemented both the paper-accurate and the repo-accurate mathematical forms (I've found that only the paper accurate form replicates Induction Heads under my setup). 
- I also ported the original CUDA kernel to JAX/OpenXLA's FFI with the help of coding agents. 
	- The original kernel with FFI was used instead of me doing a proper Pallas port as Pallas doesn't (at the time of writing) have a `cub::BlockScan` equivalent, despite JAX haivng `jax.lax.associative_scan`. The associative scan op also doesn't work as Pallas doesn't support scatter operations at the time of writing.
### Mamba-2
I've implemented *a* form of Structed State Space Duality, but turns out it's not quite what they did for the paper; my form is a bit more computationally expensive. I'm not gonna push anything until I actually have a proper SSD kernel.
### Mamba-3
Work on this hasn't started yet