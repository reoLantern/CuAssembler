# CuAssembler: An unofficial CUDA assembler

## What is CuAssembler

**CuAssembler** 是一个用于 nvidia CUDA 的非官方 assembler。它是一个 assembler：读取汇编（SASS / sass），并写出机器码（在 cubin 中）。它不是另一个 compiler，不像 nvidia 官方提供的 `nvcc`（用于 CUDA C）或 `ptxas`（用于 PTX）那样。

**CuAssembler** 的目标是弥合 `ptx`（nvidia 官方支持且有文档的最低层级表示）与机器码之间的鸿沟。一些类似的项目包括 `asfermi` 和 `maxas`，它们只能处理部分 CUDA instruction set。CuAssembler 目前支持 `Pascal/Volta/Turing/Ampere` instruction set（`SM60/61/70/75/80/86/...`），但由于大多数 instruction set 都可以通过自动探测获得，该机制也很容易扩展到更老、以及可能的未来 CUDA instruction set。

**NOTE**: 这个库仍处在早期阶段，还有大量工作要做；接口与架构可能会变化，使用风险自担。

## When and how should CuAssembler be used

很多 CUDA 用户在对 CUDA C 代码做优化之后，会用 `cuobjdump` 检查生成的 SASS（sass）代码。调优 SASS 最简单的方式，是直接修改 CUDA C 代码，然后再次检查生成的 SASS。对很多场景而言这已经足够（如果你真的很擅长 : )）。但对那些希望把优化做到“每一条指令”的 ninja 程序员来说，当你无法让 compiler 生成你想要的代码时就会很沮丧。另一种调优方法是修改中间层的 PTX，不过 PTX 充满了含糊的变量，既繁琐又难以阅读，而且最终生成的机器码也并不总是令人满意。CuAssembler 允许用户直接调优生成的 SASS。

需要强调的是：对大多数 CUDA 程序员而言，CUDA C（有时是 PTX）永远是第一选择。它功能完备，包含大量编译优化，且由 nvidia 官方支持并有文档。他们最了解自己的硬件，因此 compiler 也能做一些架构相关的优化。当生成的 SASS 远不符合预期时，你往往仍然有很大的空间可以在高级语言层面发挥；也有大量社区与论坛可供求助。与高级语言相比，玩汇编会非常痛苦：你必须关心原本应由 compiler 自动处理的一切。只有当你已经非常熟悉 CUDA C 与 PTX，尝试了各种你知道的优化技巧，但仍觉得生成的代码不够理想时，才适合把汇编作为一个可行选项。即便如此，通常也更方便先从 CUDA C 或 PTX 入手，然后在生成的 SASS 上做一些小幅修改。这正是 CuAssembler 的主要设计用途：提供一个对生成机器码做小范围调整的选项（官方工具无法做到这一点）。

CuAssembler 的另一个重要用途是做 micro-benchmarking：通过专门设计的小程序探测 micro-architecture 的细节。高质量的代码优化通常需要对硬件有相当深入的理解，尤其是性能相关指标，例如不同指令的 latency 与 throughput、cache hierarchy、各级 cache 的 latency 与 throughput、cache replacement policy 等。很多 micro-benchmarking 可以用 CUDA C 来完成，但用汇编更直接、更灵活：你不仅能任意安排指令顺序，还能直接设置 control code——这些在 CUDA C 或 PTX 中无法实现。

作为 assembler，CuAssembler 会把汇编“字面地”翻译成机器码，然后把机器码嵌入到 cubin 中，以便加载与执行。语义层面的正确性由程序员负责，例如显式 register 分配、指令的合理编排、以及寄存器的正确使用（例如 64bit 变量的 register pair 总是从偶数寄存器开始）。因此你应先熟悉这些约定，否则你根本无法写出合法的汇编代码，而这种错误也很难显眼地被发现。即使汇编语法合法，也不代表程序语义正确。CUDA 程序涉及多种资源，例如 general purpose registers、predicates、shared memory 等，它们需要与硬件配置匹配，并且能用于启动指定维度的 blocks。要严格检查整个程序的正确性，需要对 launch model 与 instruction set（语法与语义）都有全面理解，这在没有官方支持的情况下几乎不可能。因此，CuAssembler 只能提供非常有限的帮助，其余仍需用户自行保证程序正确性。

## A short HOWTO

CuAssembler 并不是为了从零开始创建 CUDA 程序而设计的，它需要与其他 CUDA toolkits 一起工作。你需要一个“起点 cubin”，它可以由 `nvcc`（从 CUDA C，使用 `-cubin` 或 `--keep`）生成，也可以由 `ptxas`（从手写或调优后的 PTX）生成。当前 `nvcc` 不支持在修改过的 cubin 基础上继续 link（由于其脆弱性，未来也不太可能支持），因此生成的 cubin 通常需要通过 driver API 加载。不过 `nvcc` 有一个 `--dryrun` 选项，可以列出真实执行的构建命令；我们可以 hack 这个流程（实际上只需要 hack “从 PTX 生成 cubin 的 `ptxas` 步骤”）。这样就可以只用 runtime API 来运行程序，使用起来更简单。但这也意味着我们方法的限制：cubin 中的所有 sections、symbols、global variables 等应尽量保持不变，否则 hack 可能无法正常工作。

请记住：在开始用 CuAssembler 编写之前，先把其他优化工作做好。因为对输入 cubin 的任何修改都可能使 CuAssembler 中的修改失效，随后你可能需要重新做一遍所有工作。

参见 [User Guide](UserGuide_CN.md) 和 [Tutorial](Tutorial_CN.md) 获取基本教程以及输入格式介绍。

### Prerequisites

* **CUDA toolkit 10+**: 需要 10+ 版本来支持 `sm_75`（Turing instruction set），Ampere 则需要 11+。实际上 CuAssembler 在把 `cubin` 保存为 `cuasm` 时只会使用独立程序 `nvdisasm`，而 `cuobjdump` 可能用于 dump SASS。如果你从 `cuasm` 开始，则不需要 CUDA toolkit。**NOTE**: 已知在某些版本里，一些指令或 modifier 不会出现在 disassembly 文本中，因此你可能需要尝试更高版本以确认问题是否已修复。由于 `nvdisasm` 与 `cuobjdump` 是独立程序，你不必安装完整 toolkit，只需这两个程序即可。 
* **Python 3.8+**: 更早的 Python 版本也许可用，但未测试。
* **Sympy 1.4+**: 需要支持任意精度的整数（或有理数）矩阵，用于求解 LAE，以及计算 `V` 的 null space。**NOTE**: 在 1.4 之前，sympy 似乎会缓存所有大整数，这在你 assemble 很多指令时可能表现得像 memory leak。
* **pyelftools**: 用于处理 cubin 文件的 ELF toolkit。

`sympy` 和 `pyelftools` 可以通过 `pip install sympy pyelftools` 安装。

### Settings and Simple Usage

**PATH** 与 **PYTHONPATH**：你可能需要把 CuAssembler 的 bin 路径（`CuAssembler/bin`）加入系统 `PATH` 以便脚本可直接运行；并且必须把仓库根目录加入 `PYTHONPATH` 才能 `import CuAsm`。你可以在 `.bashrc` 里加入如下内容（按需修改路径）：

```
  export PATH=${PATH}:~/works/CuAssembler/bin
  export PYTHONPATH=${PYTHONPATH}:~/works/CuAssembler/
```

在 `bin` 目录中，CuAssembler 提供了若干 python 脚本（`cuasm/hnvcc/hcubin/dsass/...`）来加速开发流程。直接用 `python cuasm.py` 或 `cuasm.py` 仍然不够方便，因此可以创建一个 simbol link：

```
ln -s cuasm.py cuasm
chmod a+x cuasm
```

你可以把这个 simbol link 放到当前 `PATH` 中的某个目录里，而不是把 `CuAssembler/bin` 加到系统 `PATH`。

NOTE：除 `hnvcc` 外，大多数脚本也能在 windows 下工作；`bin` 下的 `*.bat` 是命令行 wrapper。

#### cuasm

```
usage: cuasm [-h] [-o OUTFILE] [-f LOGFILE] [-v | -q] [--bin2asm | --asm2bin] infile [infile ...]

    Convert cubin from/to cuasm files.

    NOTE 1: if the output file already exist, the original file will be renamed to "outfile~".
    NOTE 2: if the logfile already exist, original logs will be rolled to logname.1, logname.2, until logname.3.

positional arguments:
  infile                Input filename. If not with extension .cubin/.bin/.cuasm/.asm, direction option --bin2asm or --asm2bin should be specified.

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --output OUTFILE
                        Output filename, inferred from input filename if not given.
  -f LOGFILE, --logfile LOGFILE
                        File name for saving the log, default to none.
  -v, --verbose         Verbose mode, showing almost every log.
  -q, --quiet           Quiet mode, no log unless errores found.
  --bin2asm             Convert from cubin to cuasm.
  --asm2bin             Convert from cuasm to cubin.

Examples:
    $ cuasm a.cubin
        disassemble a.cubin => a.cuasm, text mostly inherited from nvdisasm. If output file name is not given,
        the default name is replacing the ext to .cuasm

    $ cuasm a.cuasm
        assemble a.cuasm => a.cubin. If output file name is not given, default to replace the ext to .cubin

    $ cuasm a.cubin -o x.cuasm
        disassemble a.cubin => x.cuasm, specify the output file explicitly

    $ cuasm a.cubin x.cuasm
        same as `cuasm a.cubin -o x.cuasm`

    $ cuasm a.o --bin2asm
        disassemble a.o => a.cuasm, file type with extension ".o" is not recognized.
        Thus conversion direction should be specified explicitly by "--bin2asm/--asm2bin".

    $ cuasm a.cubin -f abc -v
        disassemble a.cubin => a.cuasm, save log to abc.log, and verbose mode
```

#### dsass

```
usage: dsass [-h] [-o OUTFILE] [-k] [-n] [-f LOGFILE] [-v | -q] infile [infile ...]

    Format sass with control codes from input sass/cubin/exe/...

    The original dumped sass by `cuobjdump -sass *.exe` will not show scoreboard control codes,
    which make it obscure to inspect the dependencies of instructions.
    This script will extract the scoreboard info and show them with original disassembly.

    CAUTION: the sass input should with exactly same format of `cuobjdump -sass`, otherwise
             the parser may not work correctly.

    NOTE 1: For cubins of sm8x, the cache-policy desc bit of some instruction will be set to 1
            to show desc[UR#] explicitly, other type of inputs(sass/exe/...) won't do the hack,
            which means some instructions may not be assembled normally as in cuasm files.
            This also implies for desc hacked sass, code of instructions may be not consistent either.

    NOTE 2: if the output file already exist, the original file will be renamed to "outfile~".
    NOTE 3: if the logfile already exist, original logs will be rolled to log.1, log.2, until log.3.

positional arguments:
  infile                Input filename, can be dumped sass, cubin, or binary contains cubin.

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --output OUTFILE
                        Output filename, infered from input filename if not given.
  -k, --keepcode        Keep code-only lines in input sass, default to strip.
  -n, --nodeschack      Do not hack desc bit, no matter SM version it is.
  -f LOGFILE, --logfile LOGFILE
                        File name for saving the logs, default to none.
  -v, --verbose         Verbose mode, showing almost every log.
  -q, --quiet           Quiet mode, no log unless errores found.

Examples:
    $ dsass a.cubin
        dump sass from a.cubin, and write the result with control code to a.dsass

    $ dsass a.exe -o a.txt
        dump sass from a.cubin, and write the result with control code to a.txt

    $ dsass a.sass
        translate the cuobjdumped sass into a.dsass

    $ dsass a.cubin -f abc -v
        convert a.cubin => a.dsass, save log to abc.log, and verbose mode

    $ dsass a.cubin -k
        usually lines with only codes in source sass will be ignored for compact output.
        use option -k/--keepcode to keep those lines.
```

#### hnvcc

**NOTE**: hnvcc 仅在 linux 下可用。

```
Usage: hnvcc nvcc_args...

hnvcc is the hacked wrapper of nvcc.
The operation depends on the environment variable 'HNVCC_OP':
    Not-set or 'none' : call original nvcc
    'dump' : dump cubins to hack.fname.sm_#.cubin, backup existing files.
    'hack' : hack cubins with hack.fname.sm_#.cubin, skip if not exist 
    Others : error

CAUTION:
    hnvcc hack/dump need to append options "-keep"/"-keep-dir" to nvcc.
    If these options are already in option list, hnvcc may not work right.

Examples:
    $ hnvcc test.cu -arch=sm_75 -o test               
        call original nvcc

    $ HNVCC_OP=dump test.cu -arch=sm_75 -o test       
        dump test.sm_#.cubin to hack.test.sm_#.cubin

    $ HNVCC_OP=hack test.cu -arch=sm_75 -o test       
        hack test.sm_#.cubin with hack.test.sm_#.cubin
```

#### hcubin

```
usage: hcubin [-h] [-o OUTFILE] [-f LOGFILE] [-v | -q] infile [infile ...]

    Hack the sm8x cubin with valid cache-policy desc bit set.

    Currently the disassembly of nvdisasm will not show default cache-policy UR:

    /*00b0*/                   LDG.E R8, [R2.64] ;                      /* 0x0000000402087981 */
                                                                        /* 0x000ea8000c1e1900 */
    /*00c0*/                   LDG.E R9, desc[UR6][R2.64+0x400] ;       /* 0x0004000602097981 */
                                                                        /* 0x000ea8200c1e1900 */

    The first disassembly line should be `LDG.E R8, desc[UR4][R2.64] ;`,
    in which UR[4:5] is the default cache-policy UR and not showed, which may cause assembly confusion.

    But if the 102th bit(the "2" in last line 0x000ea8200c1e1900) is set,
    all cache-policy UR will be showed, that will complete the assembly input for the encoding.

    This script will set that bit for every instruction that needs desc shown.

positional arguments:
  infile                Input filename, should be a valid cubin file.

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --output OUTFILE
                        Output filename, infered from input filename if not given.
  -f LOGFILE, --logfile LOGFILE
                        File name for saving the logs, default to none.
  -v, --verbose         Verbose mode, showing almost every log.
  -q, --quiet           Quiet mode, no log unless errores found.

Examples:
    $ hcubin a.cubin
        hack a.cubin into a.hcubin, default output name is replacing the ext to .hcubin

    $ hcubin a.cubin -o x.bin
        hack a.cubin into x.bin

    $ hcubin a.cubin x.bin
        same as `hcubin a.cubin -o x.bin`
```

### Classes

* **CuAsmLogger**: 基于 python logging module 的 logger。注意所有 logging 都通过私有 logger 完成，因此如果其他模块使用自己的 logger，一般不会受到影响。
* **CuAsmParser**: parser，用于解析用户修改后的 `.cuasm` 文本，并把结果保存为 `.cubin`。 
* **CubinFile**: 可以读入 `.cubin` 文件，并将其重写为可编辑的 `.cuasm` 文本。
* **CuInsAssembler**: 用于处理某个特定 instruction *key* 的 value matrix `V` 与 `w` 的求解，例如 `FFMA_R_R_R_R`。
* **CuInsAssemblerRepos**: 所有已知 *keys* 的 `CuInsAssembler` 仓库（repository）。从零构建一个可用的 repos 非常耗时，而且需要覆盖常用指令的足够多样化输入。因此提供了预先收集的 repos：`DefaultInsAsmRepos.${arch}.txt`。**Note**: repository 可能不完整，但用户可以很容易地更新它。
* **CuInsParser**: 用于把 instruction string 解析成 *keys*、*values* 与 *modifiers*。
* **CuInsFeeder**: 简单的 instruction feeder，从 `cuobjdump` dump 的 SASS 中读取指令。
* **CuKernelAssembler**: kernel 的 assembler，需要处理 kernel 范围内的参数，主要是 nvinfo attributes。
* **CuNVInfo**: 处理 cubin 中 `NVInfo` section 的简单类。该类远未完善与健壮，因此 CuAssembler 对部分 `NVInfo` attributes 的支持非常有限。
* **CuSMVersion**：为所有 SM versions 提供统一接口的类。建议其他类尽量不包含架构相关的特殊处理（至少希望如此……）。因此对于未来架构，大部分工作应该集中在该类中。  

## Future plan

更可能支持的方向：

* 更好的指令覆盖率，对官方不支持指令的 bugfix。
* 扩展到更多 compute capabilities，重点关注 `sm_60/61/70/75/80/86`。 
* 借助 `nvdisasm` 做更健壮的正确性检查。
* 自动设置 control codes。 
* 支持 alias 与 variable，以便更易编程（可能通过预处理实现）。

不太可能但仍在计划中的方向：
* Register 计数，甚至 register 分配
* 更健壮的解析与更友好的报错
* control flow 支持？也许可以通过 python 预处理实现
* 以及其他……

