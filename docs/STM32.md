# STM32 Engineering Check

The auxiliary target is an STM32F103C8T6, with a Cortex-M3 core, 64 KiB Flash,
20 KiB SRAM and a nominal 72 MHz firmware clock. The PC computes 49 by 10 MFCC
features. This backend folds batch normalization and uses packed W4/W8 weights,
A8 activations, int32 accumulation and fixed-point output scaling.

`firmware/stm32f103/` contains board, startup, linker and inference sources.
`scripts/prepare_stm32.py` exports and host-checks the integer graph. Re-export
requires the speech seed-42 checkpoint, quantization-role feature cache and an
installed Zig 0.14.1 compiler; the compiler is not redistributed here.

```powershell
python scripts/prepare_stm32.py --zig <path-to-zig-0.14.1>
```

The source retains the original portable-tool defaults. For flashing and
capture, install the documented OpenOCD build in the expected tool directory or
provide the capture script's `--openocd` argument. Read
`firmware/stm32f103/README_中文.md` and each script's parameters before accessing
hardware. Flashing replaces the board's current firmware; back it up first.

The five measured masks are 0, 2, 34, 35 and 127. Each used eight unique
quantization-role inputs, one warm-up and three timed repeats per input. Across
120 timed inferences, captured integer logits matched the host backend.
The complete firmware image was 42,360 bytes for all-W8 and 26,584 bytes for
candidate-layer W4; mean network latency was 1,850.434 and 1,923.964 ms,
respectively. Observed SRAM was 16,956 bytes in both cases.

These are single-board engineering observations. They do not measure full
test-set prediction quality, input-feature extraction time or energy. Unpacking,
control flow and memory access are possible timing contributors, not separately
profiled explanations. A different kernel library must be benchmarked rather
than assumed faster.
