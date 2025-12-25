- [使用 CUDA runtime API 的 CuAssembler 教程](#tutorial-for-using-cuassembler-with-cuda-runtime-api)
  - [从一个 CUDA C 示例开始](#start-from-a-cuda-c-example)
  - [Blitz 版本](#blitz-version)
  - [Long March 版本](#long-march-version)
    - [将 CUDA C 构建为 Cubin](#build-cuda-c-into-cubin)
    - [将 Cubin 反汇编为 Cuasm](#disassemble-cubin-to-cuasm)
    - [在 cuasm 中调整汇编代码](#adjust-the-assembly-code-in-cuasm)
    - [将 cuasm 汇编为 cubin](#assemble-cuasm-into-cubin)
    - [Hack 原始可执行文件](#hack-the-original-executable)
    - [运行或调试可执行文件](#run-or-debug-the-executable)

下面我们会通过一个简单的 `cudatest` 例子展示 CuAssembler 的基本用法。**CuAssembler** 只是一个 assembler，它的主要目的，是根据用户输入的汇编生成 cubin 文件。所有 device 初始化、数据准备、kernel 启动都需要用户自行完成（可能使用 CUDA driver API）。不过通常从 runtime API 入手更方便；本教程将演示在 CUDA runtime API 流程中使用 CuAssembler 的通用工作流。

这个教程远不完整：完成这个“看似简单”的任务仍需要大量 CUDA 基础知识。这里不会完整展示所有代码，一些常见构建步骤也会被省略，但我认为你可以理解整体思路……如果不能，那你可能来得太早了，建议先熟悉 CUDA 的基础用法再回来。

一些有用的前置知识参考：
* [CUDA](https://docs.nvidia.com/cuda/index.html) 基础知识，至少需要阅读 CUDA C programming guide。
* [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) 与 [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)：很多用户只通过 IDE 间接使用这些工具，但在这里你需要不时在命令行里直接操作它们。
* ELF Format：ELF 格式有很多参考资料，既有通用也有架构相关的，例如 [这份文档](http://downloads.openwatcom.org/ftp/devel/docs/elf-64-gen.pdf)。CuAssembler 目前只支持 **64bit** ELF（**little endian**）。
* 常见 assembly directives：`nvdisasm` 的许多约定与 gnu assembler 类似。由于 `nvdisasm` 的 disassembly 语法没有官方文档，熟悉 [Gnu Assembler directives](https://ftp.gnu.org/old-gnu/Manuals/gas-2.9.1/html_chapter/as_7.html) 会有帮助。实际上 cuasm 里只用到了很少的 directives，需要更多信息时再查即可。**NOTE**: 某些 directives 可能是架构相关的，你需要自行甄别。
* CUDA PTX 与 SASS 指令：在你能写任何汇编之前，必须先理解这门语言。目前 SASS 没有官方（至少没有全面的）文档，只有一个 [简单的 opcode 列表](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref)。熟悉 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 及其文档，会极大帮助你理解 SASS 汇编的语义。

# Tutorial for using CuAssembler with CUDA runtime API

如 [ReadMe](README_CN.md) 所述，CuAssembler 最常见的用途是将用户修改后的汇编 hack 回 cubin（官方不支持）。但在很多情况下，我们仍希望复用 nvcc 的大部分编译步骤，只对最终 cubin 做一点点修改。如果我们仍能用 CUDA runtime API 来运行被 hack 的 cubin，而不是另写一个程序用 driver API 加载它，那么会方便得多。

下面将展示一个例子：在 nvcc 的构建流程中 hack cubin。

## Start from a CUDA C example

首先我们需要创建一个 `cudatest.cu` 文件，里面包含足够的 kernel 信息。你可以从任何包含显式 kernel 定义的 CUDA samples 开始。某些 CUDA 程序并不会由用户显式编写 kernel，而是调用预编译在库中的 kernels；这种情况下你无法通过 runtime API hack cubin，你需要去 hack 那个库——那是完全不同的故事。这里我们只关注 *user kernels*，而不是 *library kernels*。一个 kernel 示例可能像这样（其他行略）：

```c++
__global__ void vectorAdd(const float* a, const float* b, float* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
```

目前 CuAssembler 并不完全支持修改 kernel args、globals（constants、texture/surface references）等，因此这些信息（size、name 等）应在 CUDA C 中定义好，并从 cubin 继承到 CuAssembler。这里的最佳实践是：先做一个朴素但能正确运行的 kernel 版本，把所有需要的资源准备好；然后在汇编层只修改指令序列。这是 CuAssembler 最稳健的使用方式。

**NOTE**: 当你进入最终的汇编调优阶段时，再去修改原始 CUDA C 往往会非常不可靠、也很容易出错；因此强烈建议保持 CUDA C 中的“其他东西”不变。如果你真的需要这么做，你可能不得不对生成的汇编做一次大规模重构。对生成的 `*.cuasm` 文件做版本控制可能会帮助你更容易（也希望更不痛苦地）走完流程。

## Blitz Version
CuAssembler 提供了一套用户工具来加速 hack cubin 的基本开发步骤；这些脚本的基本用法请参见 [ReadMe](README_CN.md) 的 “Settings and Simple Usage” 小节。借助这些脚本，hack 与恢复（resuming）可以非常快地完成。

* Step 1: 将 makefile `CuAssembler/bin/makefile` 复制到 `cudatest.cu` 所在目录。把 makefile 中的 `BASENAME` 设置为 `BASENAME=cudatest`；把 `ARCH` 设置为你的 SM 版本。如果需要额外 include 或 link，修改 `$INCLUDE` 或 `$LINK`。
* Step 2: 运行 `make d2h`。你会得到 3 个新文件：
  * `dump.cudatest.sm_75.cubin` : 从 `cudatest.cu` 编译得到的原始 cubin。
  * `dump.cudatest.sm_75.cuasm` : 原始 cubin 的 disassembly。
  * `hack.cudatest.sm_75.cuasm` : `dump.cudatest.sm_75.cuasm` 的一份拷贝，供用户修改。
* Step 3: 修改 `hack.cudatest.sm_75.cuasm`。
* Step 4: 运行 `make hack`：它会把 `hack.cudatest.sm_75.cuasm` 汇编为 `hack.cudatest.sm_75.cubin`，用这个 hacked 版本替换原始 cubin，然后继续后续构建步骤，生成最终可执行文件 `cudatest`。
* Step 5: 运行 `cudatest` 检查结果。 


## Long March Version

### Build CUDA C into Cubin

`nvcc` 是将 `.cu` 文件构建为可执行文件的标准方式，例如 `nvcc -o cudatest cudatest.cu`。但我们需要从中间产物 `cubin` 开始，因此我们会使用 `nvcc` 的 `--keep` 选项来保留所有中间文件（例如 ptx、cubin 等）。默认情况下只会生成 PTX 与 cubin 的“最低支持 SM 版本”；如果你需要特定 SM 版本的 cubin，则需要指定 `-gencode` 选项，例如 `-gencode=arch=compute_75,code=\"sm_75,compute_75\"` 用于 Turing（`sm_75`）。完整命令大概是：

```
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep
```

然后你会在中间文件目录（也可能就是当前目录）得到类似 `cudatest.1.sm_75.cubin`（编号可能不同）的 cubin 文件；这就是我们要的起点 cubin。

**NOTE**: 有时 `nvcc` 会生成多个不同版本的 cubin，并且可能为每个 SM 版本额外生成一个空的 cubin。你可以用 `nvdisasm` 查看内容，或者直接用文件大小来判断。

`nvcc` 的另一个关键信息是“完整构建步骤”。因此我们使用 `--dryrun` 来列出 `nvcc` 调用的所有步骤：

```
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --dryrun
```

你可能会得到类似下面的输出（部分行略；你的输出可能不同）：

```sh
$ nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --dryrun
...
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=55 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "cudatest.cu" -o "/tmp/tmpxft_0000016a_00000000-9_cudatest.cpp1.ii" 
#$ cicc --c++14 --gnu_version=70500 --display_error_number --orig_src_file_name "cudatest.cu" --orig_src_path_name "temp/cudatest.cu" --allow_managed   -arch compute_75 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_0000016a_00000000-3_cudatest.fatbin.c" -tused --gen_module_id_file --module_id_file_name "/tmp/tmpxft_0000016a_00000000-4_cudatest.module_id" --gen_c_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.c" --stub_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.gpu"  "/tmp/tmpxft_0000016a_00000000-9_cudatest.cpp1.ii" -o "/tmp/tmpxft_0000016a_00000000-6_cudatest.ptx"
#$ ptxas -arch=sm_75 -m64  "/tmp/tmpxft_0000016a_00000000-6_cudatest.ptx"  -o "/tmp/tmpxft_0000016a_00000000-10_cudatest.sm_75.cubin" 
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=75,file=/tmp/tmpxft_0000016a_00000000-10_cudatest.sm_75.cubin" "--image3=kind=ptx,sm=75,file=/tmp/tmpxft_0000016a_00000000-6_cudatest.ptx" --embedded-fatbin="/tmp/tmpxft_0000016a_00000000-3_cudatest.fatbin.c" 
#$ rm /tmp/tmpxft_0000016a_00000000-3_cudatest.fatbin
#$ gcc -D__CUDA_ARCH_LIST__=750 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=55 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "cudatest.cu" -o "/tmp/tmpxft_0000016a_00000000-5_cudatest.cpp4.ii" 
#$ cudafe++ --c++14 --gnu_version=70500 --display_error_number --orig_src_file_name "cudatest.cu" --orig_src_path_name "temp/cudatest.cu" --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.cpp" --stub_file_name "tmpxft_0000016a_00000000-6_cudatest.cudafe1.stub.c" --module_id_file_name "/tmp/tmpxft_0000016a_00000000-4_cudatest.module_id" "/tmp/tmpxft_0000016a_00000000-5_cudatest.cpp4.ii" 
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"   -m64 "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.cpp" -o "/tmp/tmpxft_0000016a_00000000-11_cudatest.o" 
#$ nvlink -m64 --arch=sm_75 --register-link-binaries="/tmp/tmpxft_0000016a_00000000-7_cudatest_dlink.reg.c"    "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_0000016a_00000000-11_cudatest.o"  -lcudadevrt  -o "/tmp/tmpxft_0000016a_00000000-12_cudatest_dlink.sm_75.cubin"
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=75,file=/tmp/tmpxft_0000016a_00000000-12_cudatest_dlink.sm_75.cubin" --embedded-fatbin="/tmp/tmpxft_0000016a_00000000-8_cudatest_dlink.fatbin.c" 
#$ rm /tmp/tmpxft_0000016a_00000000-8_cudatest_dlink.fatbin
#$ gcc -D__CUDA_ARCH_LIST__=750 -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_0000016a_00000000-8_cudatest_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_0000016a_00000000-7_cudatest_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=55 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/usr/local/cuda-11.6/bin/crt/link.stub" -o "/tmp/tmpxft_0000016a_00000000-13_cudatest_dlink.o" 
#$ g++ -D__CUDA_ARCH_LIST__=750 -m64 -Wl,--start-group "/tmp/tmpxft_0000016a_00000000-13_cudatest_dlink.o" "/tmp/tmpxft_0000016a_00000000-11_cudatest.o"   "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "cudatest" 
```

把这些命令保存到脚本文件里（例如 linux 下用 `*.sh`，windows 下用 `*.bat`；记得先把它们 **uncomment**）。当你想把 hacked cubin 重新嵌回可执行文件，并且像“从未发生 hack”一样运行时，就会用到它们。

### Disassemble Cubin to Cuasm

Cubin 是二进制文件，用户无法直接修改，因此我们需要先把它 disassemble。

**Command-Line Approach**：

脚本 `CuAssembler/bin/cuasm.py` 提供了一个方便的方式把 cubin 反汇编为 cuasm 文本形式。运行 `cuasm -h` 查看更多信息。

```
cuasm cudatest.sm_75.cubin
```

这会生成 disassembly 文件 `cudatest.sm_75.cuasm`，更易理解与编辑。NOTE：该 disassembly 主要继承自 `nvdisasm`，但增加了一些 CuAssembler 需要的新 directives 以便可以再汇编回去。`nvdisasm` 的原始 disassembly 不能被 CuAssembler 识别。

**Programming Approach**：
由于 CuAssembler 是一个 python package，大多数功能也可以通过编程方式调用。我们可以写一个 python 脚本来把 `cubin` disassemble 为 `cuasm`：

```python
from CuAsm.CubinFile import CubinFile

binname = 'cudatest.sm_75.cubin'
cf = CubinFile(binname)
asmname = binname.replace('.cubin', '.cuasm')
cf.saveAsCuAsm(asmname)
```

多数情况下命令行方式更顺手，但编程方式更灵活，可以支持更复杂的 pre-processing 或 post-processing。

### Adjust the assembly code in cuasm

`cuasm` 文件的大多数内容都拷贝自 `nvdisasm` 的 cubin 结果，并显式补充了一些 ELF 信息（例如 file header attributes、section header attributes、以及 disassembly 中没显示的隐式 sections，如 `.strtab/.shstrtab/.symtab`）。这些直接从 cubin 继承来的信息通常不应被修改（除非不得不修改，例如 sections 的 offset/size；这类内容会由 assembler 自动更新）。这并不意味着这些信息无法自动生成；但由于 NVIDIA 没有公开其约定，去完整探测会非常痛苦，因此保持不变会更安全也更容易。实际上，很多对这些信息的调整（例如增加一个 kernel、global 等）都可以通过修改原始 CUDA C 来实现，这既是官方支持的方式，也更可靠。

更多信息请参见 `TestData` 中的 [example cuasm](TestData/CuTest/cudatest.7.sm_75.cuasm)。

### Assemble cuasm into cubin

`*.cuasm` 文件无法被 CUDA 识别，因此需要把它再汇编回 `*.cubin` 才能使用。

**Command-Line Approach**：

把 cuasm 汇编为 cubin 也很简单：

```
cuasm cudatest.sm_75.cuasm -o new_cudatest.sm_75.cubin
```

CAUTION：`cuasm cudatest.sm_75.cuasm` 的默认输出可能会覆盖原始 `cudatest.sm_75.cubin`，因此建议使用新名字。为了避免意外覆盖，必要时 `cuasm` 会创建备份 `.cubin~`。

**Programming Approach**：

```python
from CuAsm.CuAsmParser import CuAsmParser

asmname = 'cudatest.7.sm_75.cuasm'
binname = 'new_cudatest.7.sm_75.cubin'
cap = CuAsmParser()
cap.parse(asmname)
cap.saveAsCubin(binname)
```

### Hack the original executable 

一旦你得到了 hacked cubin，把它放回可执行文件最简单的方式是“模仿”原始构建步骤的行为。观察 `nvcc` 的 `--dryrun` 输出，你会看到类似这样的步骤：

```bat
ptxas -arch=sm_75 -m64 "cudatest.ptx"  -o "cudatest.sm_75.cubin"
```

你可以删除该步骤之前的所有步骤（包括这个 `ptxas` 步骤），把你的 hacked cubin 重命名为 `cudatest.sm_75.cubin`，然后执行剩下的构建步骤。这样就能得到一个与直接运行 `nvcc` 构建非常类似的可执行文件。

有时你不需要 hack 所有 cubins：你可以自由选择 hack 一个或多个 `ptxas` 步骤，因为 `ptxas` 每次只处理一个文件。为了更方便使用，你也可以把这些步骤拷贝到 makefile 里，任何依赖文件发生变化时再执行 rebuild。你甚至可以写脚本或设置环境变量，在 hacked 版本与原始版本之间切换。

### Run or debug the executable

如果一切顺利，hacked cubin 与最终可执行文件应当与原始版本一样正常工作。但如果与原始 CUDA C 文件存在某些不匹配（例如 kernel names、kernel arg 布局、global constants、global texture/surface references），可执行文件可能无法正常运行。这也是我们在 hack cubin 之前就应该准备好这些信息的原因。另一个问题是：一些 symbol 信息会用于正确的调试，因此你也不应修改它们（symbol offsets 与 sizes 会由 assembler 自动更新）。

**NOTE**: debug 版本的 cubin 包含了过多信息（例如用于 source line 关联的 DWARF ……以及更多），对 assembler 来说处理难度很大。因此不建议在 debug cubin 上使用 CuAssembler。这也是推荐先在一个朴素但正确的 CUDA C 版本上工作的另一个原因。NVIDIA 提供了用于最终 SASS 级别调试的工具（例如 NSight VS 版本和 `cuda-gdb`），但在这一层级没有 source code correlation。

