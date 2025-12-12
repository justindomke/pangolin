**2025.12.12:** Update API docs to use Sphinx, be somewhat less chaotic.

**2025.12.05:** Update Pangolin logo (h/t Bob), update demos, update install instructions.

**2025.10.31:** Add experimental Torch backend

**2025.10.24:** Drop NumPyro (NumPyro is too unstable), replace with Blackjax backend. Create options for no / simple / numpy style broadcasting.

**2025.08.01:** Add notebooks demonstrating IR and interface.

**2025.07.31:** Various cleanup, improve printing.

**2024.10.25:** Deploy auto-generated API docs.

**2024.10.25:** Try to prevent users from using Loop variables are part of slices. (Throw a meaningful exception instead of just failing.)

**2024.10.16:** Fix bug introduced by previous update. (Old commit re-used plate names between different variables, which (apparently) has serious consequences in NumPyro.)

**2024.10.15:** Revamped NumPyro backend to try to deal better with discrete variables. Now, vmap uses `numpyro.plate` where possible. (Which it often isn't due to limitations in how NumPyro works.)  