#  A numerically workflow for fixed-point CNN deployment on FPGA — Reproducibility Repository

This repository guides the manuscript submitted to *Array* and provides the code and projects required to reproduce the main results:  
(i) MATLAB floating-point baseline, (ii) MATLAB fixed-point profiling/conversion, (iii) Vitis HLS C simulation and RTL/C co-simulation (cycle/latency reports), and (iv) Vivado synthesis/implementation (resource utilization reports).

**Repository:**  
https://github.com/charlicruz/A_numerically_workflow_for_fixed-point_CNN_deployment_on_FPGA
**Version used in the manuscript:** tag/commit: `array-r3` 

## Toolchain and dataset protocol 

- **MATLAB:** R2023b + Fixed-Point Designer 23.2
- **Xilinx tools:** Vitis HLS / Vivado v2020  
- **Target FPGA:** Artix-7 XC7A100T  
- **Dataset:** MNIST (controlled subset used in this paper: **8,000 train / 2,000 test**)  
- **MNIST reference:** see `keras_mnist` (as cited in the manuscript)

> Note: Results are reported on the fixed MNIST subset to enable fast iteration during fixed-point tuning and HLS design-space exploration.

## Repository structure (top-level)

- **`MATLAB/`**  
  MATLAB reference implementation and fixed-point tooling.
  - `MATLAB/CNN/CNN_FloatingPoint/` — floating-point baseline (run `TestMnistConv.m`)
  - `MATLAB/CNN/CNN_FixedPoint/` — fixed-point profiling/conversion (run `*_fixpt.m` and `*_wrapper_fixpt.m`)
  - `MATLAB/Conversions to C/` — optional MATLAB Coder/codegen artifacts (not required unless explicitly reproduced)
  - See: `MATLAB/CNN/CNN_FloatingPoint/README.md`, `MATLAB/CNN/CNN_FixedPoint/README.md`, `MATLAB/Conversions to C/README.md`

- **`Vitis HLS/`**  
  C/C++ HLS blocks + testbenches (`*_tb.cpp`) for C simulation and RTL/C co-simulation.
  - Convolution, pooling, FC/multipliers and top-level glue (`general*`)
  - See: `Vitis HLS/README.md`

- **`Vivado/`**  
  Vivado projects (`*.xpr`) used to generate utilization reports and optional simulations.
  - `ReLu/`, `SoftMax/`, `relu1x1/project_1/`, `softmax1x1/project_1/`
  - `TESTBENCHS/` — simulation assets (`*.wcfg`, `sim_1/`)
  - See: `Vivado/README.md`

---

## Quick reproduction (end-to-end)

### 1) MATLAB (floating-point baseline)
1. Open MATLAB and set the working directory to:
   - `MATLAB/CNN/CNN_FloatingPoint/`
2. Run:
   - `TestMnistConv.m`
3. Output:
   - Floating-point accuracy on the predefined MNIST evaluation subset used in the paper.

### 2) MATLAB (fixed-point profiling/conversion)
1. Set the working directory to:
   - `MATLAB/CNN/CNN_FixedPoint/`
2. Run the Fixed-Point Converter entry scripts:
   - `*_fixpt.m` and their `*_wrapper_fixpt.m` wrappers
3. Output:
   - Fixed-point behavior/profiling reports supporting per-layer `(W,I)` selection.

### 3) Vitis HLS (latency/cycle reports)
1. Open Vitis HLS and create/open a project pointing to sources in `Vitis HLS/`.
2. Use the provided `*_tb.cpp` testbenches:
   - Run **C Simulation** and **RTL/C Co-simulation**
3. Output:
   - HLS reports (II/latency/cycles) used for the manuscript’s timing tables.

### 4) Vivado (resource utilization)
1. Open one of the provided Vivado projects (example):
   - `Vivado/ReLu/ReLu.xpr`
2. Run **Synthesis** (and optionally **Implementation**) and generate **Report Utilization**.
3. Output:
   - LUT/FF/BRAM/DSP utilization used in the manuscript’s resource tables.

---

## Toolchain 
- MATLAB: R2024b + Fixed-Point Designer  
- Vitis HLS / Vivado: 2200 
- Target FPGA: Artix-7 XC7A100T  
- OS: Windows 10

---

## Notes on dataset protocol
The paper reports results on a controlled MNIST subset (8,000 train / 2,000 test) to enable rapid fixed-point/HLS iteration. Ensure you use the same subset protocol described in the manuscript for deterministic reproduction.

---

## Contact
For questions or issues reproducing results, please open a GitHub Issue in this repository.
