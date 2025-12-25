CuAssembler 用户指南
- [cubin 与 cuasm 格式简要说明](#a-brief-instruction-on-format-of-cubin-and-cuasm)
  - [File Structure](#file-structure)
  - [Sections and Segments](#sections-and-segments)
  - [cuasm 的基本语法](#basic-syntax-of-cuasm)
  - [Kernel text sections](#kernel-text-sections)
  - [Limitations, Traps and Pitfalls](#limitations-traps-and-pitfalls)
- [CuAssembler 的工作原理](#how-cuassembler-works)
  - [Automatic Instruction Encoding](#automatic-instruction-encoding)
  - [Special Treatments of Encoding](#special-treatments-of-encoding)
  - [Instruction Assembler Repository](#instruction-assembler-repository)

# A brief instruction on format of cubin and cuasm

**Cubin** 是一种 ELF 格式的 binary，因此它的大部分文件结构遵循 ELF 的通用约定。但其中也包含许多 CUDA 特有的特性。**Cuasm** 则是 cubin 的文本形式：用 assembly directives 显式描述了 cubin 的大部分信息。cuasm 中的大多数 directives 都遵循 `nvdisasm` 的语义（事实上，cuasm 的大部分内容也拷贝自 `nvdisasm` 输出），但为了让一些信息更清晰、更显式，CuAssembler 也引入了一些新 directives。

## File Structure

一个 ELF 文件包含 file header、若干 sections、以及零个或多个 program segments。通常 cubin 文件的组织结构如下：

* **File Header** : ELF file header 会给出一些通用信息，例如 identifier magic number、32bit/64bit、section header offset、program header offset 等。对 cubin 而言，它还会指定当前 cubin 的版本信息：virtual architecture version、SM version、toolkit version 等。
* **Section Data** : 每个 section 的数据本体。
* **Section Header** : 每个 section 的 header 信息，定义 section name、section offset、size、flags、type、extra info，以及与其他 sections 的 linkage 等。
* **Segment Header** : 每个 segment 的 header 信息，定义 sections 如何被加载。**NOTE**: 对某些 `ET_REL` 类型的 ELF，可能不存在 segment；它们可能会被 link 到另一个 cubin 以生成最终 executable。当前 CuAssembler 只测试了 `ET_EXEC` 类型的 ELF，它通常包含 3 个 segments。

## Sections and Segments

下面是一个 cubin sections 布局示例（debug 版本的 cubin 会包含更多 sections，这里不讨论）：

```
Index Offset   Size ES Align        Type        Flags Link     Info Name
    1     40    418  0  1            STRTAB       0    0        0 .shstrtab
    2    458    783  0  1            STRTAB       0    0        0 .strtab
    3    be0    450 18  8            SYMTAB       0    2       22 .symtab
    4   1030    5c8  0  1          PROGBITS       0    0        0 .debug_frame
    5   15f8    21c  0  4         CUDA_INFO       0    3        0 .nv.info
    6   1814     78  0  4         CUDA_INFO       0    3       1a .nv.info._Z7argtestPiS_S_
    7   188c     4c  0  4         CUDA_INFO       0    3       1b .nv.info._Z11shared_testfPf
    8   18d8     60  0  4         CUDA_INFO       0    3       1c .nv.info._Z11nvinfo_testiiPi
    9   1938     4c  0  4         CUDA_INFO       0    3       1d .nv.info._Z5childPii
    a   1984     5c  0  4         CUDA_INFO       0    3       1e .nv.info._Z10local_testiiPi
    b   19e0     4c  0  4         CUDA_INFO       0    3       1f .nv.info._Z4test6float4PS_
    c   1a30     d0  8  8    CUDA_RELOCINFO       0    0        0 .nv.rel.action
    d   1b00     70 10  8               REL       0    3       1a .rel.text._Z7argtestPiS_S_
    e   1b70     30 18  8              RELA       0    3       1a .rela.text._Z7argtestPiS_S_
    f   1ba0     40 10  8               REL       0    3       1d .rel.text._Z5childPii
   10   1be0     40 10  8               REL       0    3       14 .rel.nv.constant0._Z7argtestPiS_S_
   11   1c20     d0 10  8               REL       0    3        4 .rel.debug_frame
   12   1cf0    141  0  4          PROGBITS       2    0        0 .nv.constant3
   13   1e34     48  0  4          PROGBITS       2    0       1a .nv.constant2._Z7argtestPiS_S_
   14   1e7c    188  0  4          PROGBITS       2    0       1a .nv.constant0._Z7argtestPiS_S_
   15   2004    170  0  4          PROGBITS       2    0       1b .nv.constant0._Z11shared_testfPf
   16   2174    170  0  4          PROGBITS       2    0       1c .nv.constant0._Z11nvinfo_testiiPi
   17   22e4    16c  0  4          PROGBITS       2    0       1d .nv.constant0._Z5childPii
   18   2450    170  0  4          PROGBITS       2    0       1e .nv.constant0._Z10local_testiiPi
   19   25c0    178  0  4          PROGBITS       2    0       1f .nv.constant0._Z4test6float4PS_
   1a   2780    d80  0 80          PROGBITS       6    3 18000023 .text._Z7argtestPiS_S_
   1b   3500    200  0 80          PROGBITS  100006    3  c000029 .text._Z11shared_testfPf
   1c   3700    100  0 80          PROGBITS       6    3  a00002a .text._Z11nvinfo_testiiPi
   1d   3800    280  0 80          PROGBITS       6    3  e00002b .text._Z5childPii
   1e   3a80    180  0 80          PROGBITS       6    3  d00002c .text._Z10local_testiiPi
   1f   3c00    480  0 80          PROGBITS       6    3  a00002d .text._Z4test6float4PS_
   20   4080     5c  0  8          PROGBITS       3    0        0 .nv.global.init
   21   40e0      0  0 10            NOBITS       3    0       1a .nv.shared._Z7argtestPiS_S_
   22   40e0     a0  0  4            NOBITS       3    0        0 .nv.global
   23   40e0   1010  0 10            NOBITS       3    0       1b .nv.shared._Z11shared_testfPf
   24   40e0      0  0 10            NOBITS       3    0       1c .nv.shared._Z11nvinfo_testiiPi
   25   40e0      0  0 10            NOBITS       3    0       1d .nv.shared._Z5childPii
   26   40e0      0  0 10            NOBITS       3    0       1e .nv.shared._Z10local_testiiPi
   27   40e0      0  0 10            NOBITS       3    0       1f .nv.shared._Z4test6float4PS_
```

* `.shstrtab/.strtab/.symtab` : section string、symbol string、以及 symbol entries 的表。当前它们都直接拷贝自原始 cubin。
* `.nv.info.*` : 与 kernels 相关的一些 attributes。`cuobjdump -elf *.cubin` 可以以更可读的形式展示这些信息。当 kernel text 发生变化时，其中部分 attributes 需要更新；部分可由 CuAssembler 完成（例如 `EIATTR_EXIT_INSTR_OFFSETS`、`EIATTR_CTAIDZ_USED` 等），但仍有更多无法自动完成。一些 attributes 与某些指令的 offset 强相关；CuAssembler 使用一种特殊形式的 label 来处理这类 attributes，以便在指令序列变化后仍能工作。
* `.rel.*` : relocations。relocation section 需要与其关联的 section 配合使用，例如 `.rel.abc` 对应 `.abc`。relocation 是一种特殊机制，允许对编译期未知的一些 symbols 做 runtime 初始化，例如某些 global constants 与 function entries。
* `.nv.constant#.*` : constant memory 内容，可能是 global constants，也可能是 kernel 相关 constants。constant memories 的实际布局可能随 SM 版本（甚至 toolkit 版本？）变化，因此最好参考 CUDA C 生成的原始 SASS。在上面的例子里，constant bank 3 `.nv.constant3` 是 global（通过 `c[0x3][###]` 引用）。bank 2 是 compiler 生成的 constants；bank 0 则用于 kernel arguments 与 grid/block 常量。bank 0 与 bank 2 都是 kernel 相关的。
* `.text.*` : kernel 指令 sections。多数修改都发生在这些 sections 上。
* `.nv.shared.*` : NOBITS sections。目前我没有发现 shared memory 可以在 runtime 初始化，因此它们看起来只用于空间分配。

## Basic syntax of cuasm
cuasm 的大多数语法遵循 `nvdisasm` 的约定；但由于 `nvdisasm` 并不会展示 cubin 的全部信息，我们需要更多语法来更精确地描述文件。

**Comments**:

支持 C 风格注释 `/* ... */` 与 C++ 风格注释 `// ...`。一种特殊的 branch target annotation `(* ... *)` 也会被当作注释处理。它们都会被替换为空格。**NOTE**: 目前不支持跨行注释，所有注释必须在同一行内。

**Directives**:
directive 是预定义关键字，通常以点 `.` 开头。当前支持的、由 `nvdisasm` 定义的 directives 列表如下：

| Directive          | Notes          |
|--------------------|----------------|
| `.headerflags`*     | 设置 ELF header |
| `.elftype`*         | 设置 ELF type |
| `.section`*         | 声明一个 section |
| `.sectioninfo`*     | 设置 section info |
| `.sectionflags`*    | 设置 section flags |
| `.sectionentsize`*  | 设置 section entsize |
| `.align`           | 设置 alignment |
| `.byte`            | 输出 bytes |
| `.short`           | 输出 shorts |
| `.word`            | 输出 word (4B?) |
| `.dword`           | 输出 dword (8B?) |
| `.type`*           | 设置 symbol type |
| `.size`*           | 设置 symbol size |
| `.global`*          | 声明 global symbol |
| `.weak`*            | 声明 weak symbol |
| `.zero`            | 输出若干个 0 bytes |
| `.other`*           | 设置 symbol other  |

标有星号的 directives 当前并不真正生效，因为它们的内容会直接从原始 cubin 拷贝。CuAssembler 还定义了一些新的内部 directives（以 `.__` 为前缀），用于保持这些信息不变。

|  |                                |
|--|--------------------------------|
| ELF header | |
| | .__elf_ident_osabi |
| | .__elf_ident_abiversion |
| | .__elf_type |
| | .__elf_machine |
| | .__elf_version |
| | .__elf_entry |
| | .__elf_phoff |
| | .__elf_shoff |
| | .__elf_flags |
| | .__elf_ehsize |
| | .__elf_phentsize |
| | .__elf_phnum |
| | .__elf_shentsize |
| | .__elf_shnum |
| | .__elf_shstrndx |
| Section header | |
| | .__section_name |
| | .__section_type |
| | .__section_flags |
| | .__section_addr |
| | .__section_offset |
| | .__section_size |
| | .__section_link |
| | .__section_info |
| | .__section_entsize |
| Segment header | |
| | .__segment |
| | .__segment_offset |
| | .__segment_vaddr |
| | .__segment_paddr |
| | .__segment_filesz |
| | .__segment_memsz |
| | .__segment_align |
| | .__segment_startsection |
| | .__segment_endsection |

**Labels and Symbols**:

**label** 只是一个标识符（可包含 `.`, `$` 与任意 word character）后跟冒号 `label:`，例如：

```
  _Z10local_testiiPi:
  .L_203:
  __cudart_i2opi_f:
  $str:
  $_Z7argtestPiS_S_$_Z2f1ii:
```

label 可用于引用，以便在需要填充真实 offset 时使用。

**symbol** 是一种特殊 label，它可能对外可见（externally visible），也就是说当 module 被加载时，它对应的地址会作为 symbol 的地址。symbol 可以像这样定义：

```asm
.global         _Z10local_testiiPi
.type           _Z10local_testiiPi,@function
.size           _Z10local_testiiPi,(.L_203 - _Z10local_testiiPi)
.other          _Z10local_testiiPi,@"STO_CUDA_ENTRY STV_DEFAULT"
_Z10local_testiiPi:
```

最后一行实际上是一个 label（与 symbol 使用 **相同标识符**），用来指定该 symbol 的位置。没有对应 label 的 symbol 通常是外部定义的，例如 `vprintf` 以及一些内部 non-inline device functions。每个 symbol 在 `.symtab` section 中都有一个 entry。`cuobjdump -elf *.cubin` 可以以更可读的形式展示这些 entries。

**CAUTION**:
1. 不同类型的 symbols 需要做的处理非常多。我不想去复刻那些繁琐、甚至可能带来麻烦的处理（以及 NVIDIA 私下定义的、可能隐藏的约定）。由于大多数 symbols 都可以由 CUDA C 准备，我这里只是从原始 cubin 拷贝它们，并确保这些语句在语法上合法，但不一定真正生效。
2. `ET_REL` 类型的 ELF cubin 可能会有更多类型的 symbol（也许用于之后的 link？）。要完整支持它们非常困难，因此 CuAssembler 不支持 `ET_REL` 类型的 ELF。

## Kernel text sections

对 CuAssembler 来说，kernel text sections 是最常被修改的部分。这里我们用一个简单 kernel 展示 `cuasm` 的一些基本约定：

```c++
__constant__ int C1[11];       // C1 will be stored in constant memory
__device__ int GlobalC1[7];    // GlobalC1 will be stored in device memory (RW), loaded with relocated address
__global__ void simpletest(const int4 VAL, int* v) // contents of VAL and address of v will be stored in constant memory
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int a = v[idx]*VAL.x + GlobalC1[idx%16];

    // SHFL is an instruction needs an associated .nv.info attribute.
    a = __shfl_up_sync(0xffffffff, a, 1);  

    if (VAL.z > 0) // predicated statement
        a += C1[VAL.y]; 
    v[idx] = a;
}
```

对应的 `cuasm` 代码片段如下（由 CUDA toolkit 11.1、SM_75 生成）：

```
// --------------------- .text._Z10simpletest4int4Pi      --------------------------
.section	.text._Z10simpletest4int4Pi,"ax",@progbits
.__section_name         0x35b 	// offset in .shstrtab
.__section_type         SHT_PROGBITS
.__section_flags        0x6
.__section_addr         0x0
.__section_offset       0x4500 	// maybe updated by assembler
.__section_size         0x200 	// maybe updated by assembler
.__section_link         3
.__section_info         0xc000030
.__section_entsize      0
.align                128 	// equivalent to set sh_addralign
  .sectioninfo	@"SHI_REGISTERS=12"
  .align	128
        .global         _Z10simpletest4int4Pi
        .type           _Z10simpletest4int4Pi,@function
        .size           _Z10simpletest4int4Pi,(.L_228 - _Z10simpletest4int4Pi)
        .other          _Z10simpletest4int4Pi,@"STO_CUDA_ENTRY STV_DEFAULT"
_Z10simpletest4int4Pi:
.text._Z10simpletest4int4Pi:
    [B------:R-:W-:Y:S08]         /*0000*/                   MOV R1, c[0x0][0x28] ;
    [B------:R-:W0:-:S01]         /*0010*/                   S2R R2, SR_CTAID.X ;
    [B------:R-:W-:-:S01]         /*0020*/                   UMOV UR4, 32@lo(GlobalC1) ;
    [B------:R-:W-:-:S01]         /*0030*/                   MOV R9, 0x4 ;
    [B------:R-:W-:-:S01]         /*0040*/                   UMOV UR5, 32@hi(GlobalC1) ;
    [B------:R-:W0:-:S01]         /*0050*/                   S2R R3, SR_TID.X ;
    [B------:R-:W-:-:S01]         /*0060*/                   MOV R4, UR4 ;
    [B------:R-:W-:-:S02]         /*0070*/                   IMAD.U32 R5, RZ, RZ, UR5 ;
    [B0-----:R-:W-:Y:S05]         /*0080*/                   IMAD R2, R2, c[0x0][0x0], R3 ;
    [B------:R-:W-:Y:S04]         /*0090*/                   SHF.R.S32.HI R3, RZ, 0x1f, R2 ;
    [B------:R-:W-:Y:S04]         /*00a0*/                   LEA.HI R3, R3, R2, RZ, 0x4 ;
    [B------:R-:W-:Y:S05]         /*00b0*/                   LOP3.LUT R3, R3, 0xfffffff0, RZ, 0xc0, !PT ;
    [B------:R-:W-:-:S02]         /*00c0*/                   IMAD.IADD R7, R2.reuse, 0x1, -R3 ;
    [B------:R-:W-:Y:S04]         /*00d0*/                   IMAD.WIDE R2, R2, R9, c[0x0][0x170] ;
    [B------:R-:W-:Y:S04]         /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;
    [B------:R-:W2:-:S04]         /*00f0*/                   LDG.E.SYS R0, [R2] ;
    [B------:R-:W2:-:S01]         /*0100*/                   LDG.E.SYS R5, [R4] ;
    [B------:R-:W-:Y:S04]         /*0110*/                   MOV R6, c[0x0][0x168] ;
    [B------:R-:W-:Y:S12]         /*0120*/                   ISETP.GE.AND P0, PT, R6, 0x1, PT ;
    [B------:R-:W-:Y:S06]         /*0130*/               @P0 IMAD R6, R9, c[0x0][0x164], RZ ;
    [B------:R-:W0:-:S01]         /*0140*/               @P0 LDC R6, c[0x3][R6] ;
    [B--2---:R-:W-:Y:S08]         /*0150*/                   IMAD R0, R0, c[0x0][0x160], R5 ;
.CUASM_OFFSET_LABEL._Z10simpletest4int4Pi.EIATTR_COOP_GROUP_INSTR_OFFSETS.#:
    [B------:R-:W0:-:S02]         /*0160*/                   SHFL.UP PT, R7, R0, 0x1, RZ ;
    [B0-----:R-:W-:Y:S08]         /*0170*/               @P0 IMAD.IADD R7, R7, 0x1, R6 ;
    [B------:R-:W-:-:S01]         /*0180*/                   STG.E.SYS [R2], R7 ;
    [B------:R-:W-:-:S05]         /*0190*/                   EXIT ;
.L_20:
    [B------:R-:W-:Y:S00]         /*01a0*/                   BRA `(.L_20);
.L_228:
```

下面是一些解释：

* `.section	.text._Z10simpletest4int4Pi,"ax",@progbits` 声明一个 section，并指定 name/flags/type。`_Z10simpletest4int4Pi` 是 `void simpletest(const int4 VAL, int* v)` 的 **mangled** 名称；你可以用 `c++filt` 或 `cu++filt` 进行 demangle 还原。如果你不想要 name mangling，可以用 `extern "C"` 包裹声明（实际上非常不推荐）。
* `.__section_*` directives：内部 directives，用于定义 section header attributes。NVIDIA 似乎有一些自己的内部 flags；用户一般不关心这些，因此通常保持不变。
* `.align 128`：把当前 section 的 alignment 设为 128B。这意味着如果该 section 的 offset 不是 128B 的倍数，上一个 section 可能需要 padding。
* `.sectioninfo	@"SHI_REGISTERS=12"`：设置当前 kernel 使用的 register 数量。**NOTE**: 对 Turing 与 Ampere，不知何故会额外占用 2 个 GPR。因此如果你的 kernel 中使用的最大 GPR 编号是 `R20`，你需要设置 `@"SHI_REGISTERS=23"`（GPR 从 0 开始编号，`R20` 表示用了 21 个，再加 2 个额外占用，总计 23）。最大 GPR 编号是 255（`R255` 的编码被 `RZ` 占用），因此 kernel text 中可用的最大 GPR 编号是 `R252`。
* 对使用 block-wide barriers（例如 `__syncthreads()`）的 kernels，可能还需要另一个 attribute 指定 barrier 数量，例如 `.sectionflags @"SHF_BARRIERS=1"`。当前一个 kernel 最多可使用 16 个 barriers。
* `.global _Z10simpletest4int4Pi`：为当前 kernel 定义一个 symbol。它既用于 function exporting（global symbol 对外可见，这意味着可通过 driver API `cuModuleGetFunction` 访问），也可能用于调试（参见 `.debug_frame` section）。如前所述，symbols 通常保持不变。
* `[B------:R-:W-:Y:S08]         /*0000*/                   MOV R1, c[0x0][0x28] ;`：这是 instruction 行的规范形式：control code、十六进制注释地址、以及指令汇编字符串。control codes 的文本形式与 [Scott Gray 的 maxas](https://github.com/NervanaSystems/maxas/wiki/Control-Codes) 略有不同。在这里，control codes 被拆分为 6 个 field，并用冒号 `:` 分隔：
    - **Reuse flags**（Deprecated!!! reuse 会根据汇编文本中的 `.reuse` modifier 来设置）：4bit reuse flags 表示当前 slot 的 GPR 值会被后续指令再次读取。至少有 3 个（可能 4 个？我从未见过第 4 bit 被置位……）reuse caches slot；每一位用 `R` 表示 reuse、用 `-` 表示 none。reuse caches 似乎只对 ALU 指令生效，并且每个 slot 对应一个 operand 位置，该 operand 会带 `.reuse` 后缀（**NOTE**: CuAssembler 不会关心指令字符串中的 `.reuse` 后缀，只有 control codes 部分的序列起作用）。但也能观察到一些不一致，例如：
  
      >  [-R--:B------:R-:W-:-:S02]         /*09c0*/                   IABS R7, R5.reuse ;
      
    使用 GPR reuse 不仅有助于缓解 register bank conflict，也可能降低一些功耗。

    - **Barrier on scoreboard**：有 6 个 scoreboards（在 maxas 中称为 *dependency barrier*），编号为 0~5（maxas 中为 1~6），也与 `DEPBAR` 的 scoreboard operands（例如 `SB0`、`SB5`）一致。该 barrier field 有 6 bits，每个 scoreboard 对应 1 bit。**NOTE**: 与 maxas 展示聚合数字不同，这里把每一 bit 都展开显示：用 scoreboard 编号表示 wait，用 `-` 表示 no-wait，以便更直观地检查与指令设置的 scoreboards 的对应关系。可以同时 barrier 多个 scoreboards，例如 `B01--4-` 表示等待 scoreboards `0,1,4` 全部清零。
    - **Set scoreboard for reading**：`R#`，设置一个 scoreboard（数字编号）来“持有/标记”某些 source GPR operands 的内容，通常用于 memory 指令。
    - **Set scoreboard for writing**：`W#`，设置一个 scoreboard（数字编号）来阻止在 destination GPR 就绪之前去读取它。它用于 variable latency 指令，例如 memory load、double precision 指令、transcendental function 指令、S2R 指令等。scoreboard 依赖不仅可以通过 control codes 的 barrier field 来解决，也可以通过 `DEPBAR` 指令解决。
    - **Yield**：是否尝试让出（yield）给另一个 warp。该 bit 在不同指令上下文中含义可能不同。`Y` 表示尝试 yield 给另一个 eligible warp，`-` 表示不 yield。
    - **Stall count**：让 instruction issue 停顿一定数量的 clock cycles（十进制数字，0~15）。yield field 与 stall count field 在不同指令中的含义可能不同；由于没有公开的官方信息，这一点并不完全明确。
  
* 一个特殊 label：`.CUASM_OFFSET_LABEL._Z10simpletest4int4Pi.EIATTR_COOP_GROUP_INSTR_OFFSETS.#`：对每个 kernel，会有一些关联的 NVINFO attributes。`OFFSETS` 类型的 attributes 会为某些特殊指令生成。由于这类指令的规则尚不完整，CuAssembler 目前仍无法处理其中一部分。因此我们引入一种特殊 offset label，形式为 `.CUASM_OFFSET_LABEL.{KernelName}.{AttrName}.#`；该 offset 会被追加到对应 NVINFO attributes 列表中。对于从 cubin 生成的 cuasm，这些 labels 会自动追加。这一处理在 NVINFO 支持不完善、但又不想手工编辑 NVINFO section 时，可以减少一些手工工作量。

一些可能有用的 CUDA SASS 约定：
* Integer immediate 总是用十六进制表示。也就是说，`0x1` 表示整数 1，而 `1` 表示浮点数 1（精度取决于 opcode）。
* local labels 通常是 `.L_###`；global labels（symbols）通常在 `.symtab` 中有名称。local label 的地址可以由 assembler 直接获得并填充，例如当 `.L0=0x1000` 时，`BRA ``.L0` 会被翻译成 `BRA 0x1000`。但 symbol 的地址可能需要 relocation，由 program loader 设置；例如 `MOV R2, 32@lo(flist) ;` 或 `CALL.REL.NOINC R6` ``(_Z7argtestPiS_S_) ;`` 会被 assembler 填为 0，并在对应 section 里生成 relocation entries；最终地址会由 program loader 填充。
* 作为 ALU operands 的 constant memory 不支持 GPR indexing，只允许用于 `LDC` 指令。
* 以及更多……

## Limitations, Traps and Pitfalls

* 指令 parser 并不够健壮，一些语法错误无法被识别。
* 对各类值没有做 range check，例如 GPR index、barrier index、scoreboard index、float immediates，尤其是 integer immediates（作为 ALU operands、memory offsets 等）。
* 一些指令对 operands 有额外限制，例如 64bit GPR address 必须从偶数 GPR index 开始、某些类型的地址必须对齐等。目前这些正确性由用户自行保证。
* modifier 组合可能存在一些隐藏规则，也就是说 modifiers 可能并非完全独立工作。但我们没有这些规则的列表，因此把这部分工作留给用户。
* section info 与 symbols 不可修改（未来可能支持 append……）。原因前面多次提到：尽量保持所有隐藏约定不变，用 CUDA C 生成这些信息。

# How CuAssembler works

## Automatic Instruction Encoding

assembler 的大部分工作在于对 instruction 做 encoding。对 Turing，每条 instruction 是 128bit，在 `cuobjdump` dump 的 SASS 中会被拆成两行 64bit，例如：

```
    /*1190*/    @P0 FADD.FTZ R13, -R14, -RZ ;    /* 0x800000ff0e0d0221 */
                                                 /* 0x000fc80000010100 */
```

下面说明 CuAssembler 如何称呼这些字段：`/*1190*/` 是 instruction 的 *address*（十六进制）。`@P0` 是 *predicate*（更具体地说是 guard predicate，如 PTX 文档所述）。`FADD` 是操作类型（称为 *opcode*；严格来说 opcode 往往是编码该操作的 code field，而 `FADD` 是 opcode 的 mnemonics，两者在语境中常被互换使用）：单精度浮点加法。`.FTZ` 是 `FADD` 的一个 *modifier*，表示当输入或输出是 denormal 时 flush-to-zero。`R13`、`-R14`、`-RZ` 是 `FADD` 的 *operands*，语义是 `R13 = (-R14) + (-RZ)`。`RZ` 是一个总是产生 0 的寄存器。*modifier* 不仅针对 opcode，也包括任何能修改 operands 原始语义的东西，例如负号 `-` 或绝对值 `|*|`。

每个 field 都会编码 instruction 的一些 bits。三个 operands（`R13`、`-R14`、`-RZ`）都是 register 类型，因此这些字段不仅依赖内容，也依赖出现的位置。于是最终 code 可以写成各字段编码之和：

>`c = c("@P0") + c("FADD") + c("FTZ") + c("0_R13") + c("1_-R14") + c("2_-RZ")`。

operand `-R14`、`-RZ` 中的负号也可以视为负号 modifier `"Neg"`。在 Turing 中，`RZ` 总是 `R255` 的 alias。其他任何 operand modifiers（目前已知：predicate not 的 `!`、数值负号 `-`、数值绝对值 `|`、按位取反 `~`、以及一些 bit field 或 type specifiers 如 `.H0_H0`、`.F32` 等）也都会被剥离为独立字段。因此 code 变为：

>`c = c("@P0") + c("FADD") + c("FTZ") + c("0_R13") + c("1_Neg") + c("1_R14") + c("2_Neg") + c("2_R255")`。

现在问题变成：如何 encode 这些“元素级字段”。我们把每个字段的 encoding 拆成两部分：`Code = Value*Weight`，其中 `Value` 只依赖内容，`Weight` 只依赖该元素出现的位置（包括 operands 及其 modifiers）。

对 Turing 架构，我们把元素级 operands 分成以下几类，并为每类定义一些 values，并用一个 *label* 标识 operand 类型：

* **Indexed type**：indexed type 是一个类型前缀加正整数 index，例如 registers `R#`、predicates `P#`、uniform registers 与 predicates `UR#`/`UP#`、convergence barriers `B#`、scoreboards `SB#`。其 value 就是 index，label 就是前缀。
* **Address**：方括号中的 memory address，如 `[0x####]`。方括号内也可能包含由寄存器指定的 offset：`[R#+0x####]`，甚至更复杂：`[UR#+R#.X16+0x####]`。address 的 value 可以是一个 list，包含寄存器与 offset 的值。label 以 `A` 开头，后跟括号内成分的 labels，例如 `R` 表示只有 register，`RI` 表示 register+immediate offset，`I` 表示只有 immediate。例如 `[UR5+R8.X16+0x10]` 的 value list 为 `[5, 8, 16]`，label 为 `AURRI`，其中 `.X16` 会被剥离到 modifiers。当前如果未出现 immediate offset，所有 address 都会补上隐式 immediate offset `0`；即使某些指令不支持该形式，这也无害，因为 value 为 0 时编码贡献应为 0。另一个注意点是：某些 modifiers 可能与 values 搭配出现，例如 `[UR5.U64+R8.U32]`，因此我们会把 modifiers 标注其关联类型：`UR.U64` 与 `R.U32`，以避免任何可能的歧义（如果存在的话）。
* **Constant memory**：constant memory `c[0x##][0x####]`，第一个括号是 constant memory bank，第二个括号是 memory address。其 value 是 constant bank 与第二个括号内 address value 的 list。label 是 `cA` 加上第二个括号的 address label。
* **Integer immediate**：整数 immediate，例如 `0x0`（**NOTE**: integer immediate 必须用十六进制，裸写 `1` 或 `0` 会被当作 float immediates）。value 就是该整数的 bit representation。NOTE：负号应被视为 modifier，因为我们不知道该值会占用多少 bits。label 为 `II`。
* **Float immediate**：浮点 immediate，例如 `5`、`2.0`、`-2.34e-3`。float immediate 的 value 是其二进制表示，取决于精度（32bit 或 16bit；尚未发现 64bit）。也可能出现 denormals，如 `+Inf`、`-Inf` 与 `QNAN`。label 为 `FI`。
* **Scoreboard Set**：仅用于 `DEPBAR` 指令，设置需要等待的一组 scoreboards，例如 `{1,3}`。当前有 6 个 scoreboards，每个值对应 1bit。control codes 中等待的 scoreboard 计数应为 0，但 `DEPBAR` 能等待一个非零计数的 scoreboard。例如对 scoreboard 5 发送了 8 个请求，每个请求会递增 SB5，每个完成请求会递减 SB5。如果我们只需要其中 3 个完成，则只需等待 scoreboard 降到 `8-3=5`，可用 `DEPBAR.LE SB5, 0x5 ;` 实现。**NOTE**: 花括号中的逗号会影响 operand 的拆分，因此在解析阶段 scoreboard sets 会被转换为 indexed type `SBSET#`。label 为 `SBSET`。
* **Label**：以上未包含的其他类型。通常是字符串，例如 `SR_TID.X`、`SR_LEMASK`、`3D`、`ARRAY_2D` 等。label 的 value 很像 modifier，它的 value 依赖上下文；通常我们把 value 设为 **1**，让 weight 承担真正的编码。其 label 就是它本身。

这样我们就能得到示例指令 `@P0 FADD.FTZ R13, -R14, -RZ` 的 value list：

>`V = [0, 1, 1, 13, 1, 14, 1, 255]`,

以及待求的 weight list：

>`w = [w("@P0"), w("FADD"), w("FTZ"), w("0_R13"), w("1_Neg"), w("1_R14"), w("2_Neg"), w("2_R255")]`

有趣的是：如果我们用 `cuobjdump` dump 指令，那么每条指令的 value lists 都可以直接获得，并且答案 `c = v.*w` 也已经知道！只要我们能收集到足够多同类型指令，我们就能用线性代数方程 `c = V*w` 解出 `w`！

那么“同类型指令”指什么？理论上，你总可以把所有 values 都设为单一的 `1`，把所有 modifiers 合并为一个，然后用 weight 直接当作 code！但这样你只能 assemble 你字典里出现过的指令；它不仅需要太大空间，也有太多缺点。我们需要寻找一种模式：尽可能最大化通用性，同时尽量减少对“已知指令输入”的依赖。

对每条 instruction，values 的最小长度是 2（一个用于 predicate，一个用于 opcode，如 `FADD`、`IMAD`、`BRA`）再加上“非固定值”的 operands 数量（也就是不是 labels 的 operands）。因此我们把具有相同操作、且 operands 的数量与类型相同的指令放到同一类，并用下划线连接来标记，例如 `FFMA_R_R_R_R`、`IMAD_R_R_II_R`、`FADD_R_R_cAI`；这称为该类指令的 **Key**。随后我们可以收集该类所有已知指令编码来解出该 **Key** 对应的未知 weights。

只要你能收集到你需要的所有指令，解 weights 通常是很容易的。但遗憾的是，这并不总能做到。如果我们没有收集到足够多指令，`V` 会变成矩形矩阵，无法求出 `w` 的每个元素。但幸运的是：并非一定需要所有 weights 都已知！我们只需要确保待组装指令的 value list `v` 可以表示为 `V` 的行的线性组合。这等价于检查 `v` 是否落在 `V` 的 null space 中。有趣的是：即便 `V*w = c` 有无穷多解，但任意一个解都会给出相同的 assembled code：`v.*w`！

这也提示了 `V` 的构造方式：由于对任意 key 可以应用无限多 modifiers，values 的长度一开始通常未知，因此 value matrix `V` 的尺寸可能需要增量更新。当加入新指令时，先检查是否出现新 modifier；如果没有，再检查其 value 是否在 `V` 的 null space；如果仍不满足，再相应更新 `V`。

## Special Treatments of Encoding

上面的框架试图最大化通用性，同时把所需工作量降到最低。目前我们发现它在 Turing 上运行良好，并且相信它也应该适用于更早以及可能的未来 CUDA instruction sets。

不过，虽然 CuAssembler 尽量与 `cuobjdump` 的汇编约定保持一致，但这门复杂语言由 nvidia 定义而非我们，因此不可避免会有一些例外无法适配我们的简单框架：

* **PLOP3**：`PLOP3` 的 `immLut` bits 并不连续。例如 `PLOP3.LUT P2, PT, P3, P2, PT, 0x2a, 0x0 ;` 中，`immLut` 为 `0x2a = 0b00101010`，编码却形如 `0b 00101 xxxxx 010`，中间夹了另外 5 bits。因此该 operand 会被特殊处理：提前把 bits 拆开；而 `LOP3` 看起来没问题。
* **I2F, F2I, I2I, F2F for 64bit types**：64bit 数据类型转换的 opcode 与 32bit 不同。但 32bit 的 modifier 不会显式展示，因此像 `F64` 这样的 modifier 无法同时表达 `F32` 与 `F64` 的差异以及 opcode 的变化。对此我们添加了一个新 modifier `CVT64`，让它与 `F64` 搭配工作。
* **BRA, BRX, JMP, CALL, RET, etc.**：所有 branch/jump 类型指令都有一个 target address operand。但在真实 encoding 中，它们需要知道当前指令的地址，并且 operand 实际上是 *relative offset*。问题在于 relative offset 可能为负，这需要另一个 modifier 来探测应使用多少 bits。目前我们简单地修改了 target address operand，并在需要时添加 negative modifier。
* **I2I, F2F, IDP, HMMA, etc.**：一些指令存在 position-dependent modifiers，例如 `F2F.F32.F64` 与 `F2F.F64.F32` 不同。我们在每个 modifier 后追加一个后缀 `@idx`，以便区分它们。该方法只对“这类 modifiers 数量固定”的指令有效（不包含 operand modifiers）。仍存在一些指令具有可变数量的 modifiers，并且包含 position-dependent modifiers，例如 `HMMA` 与 `IDP`；仍在研究中……

由于这些特殊处理，CuAssembler 理论上应该可以把 `cuobjdump` dump 出来的 SASS 全部 re-assemble 回完全一致的 code。但总会有例外——至少目前我们知道一个：当前唯一无法从 `cuobjdump` re-assemble 的指令类型是：

>`FSEL R5, R5, +QNAN , P0 ;`

在我们的处理里，`+QNAN` 是 float immediate，但它的 bit representation 并非 *UNIQUE*：IEEE 754 里存在一类 `+QNAN`，它们的 exponent 相同但 significand 可以是任意非零值。这里 `FSEL` 似乎把寄存器设置为某个特定二进制，而不是“任意 `+QNAN`”。由于该信息并不包含在指令本身，无法恢复。对此我们提供另一种方式来表示 float immediates：显式写出每一 bit，例如 `0F3f800000`，类似 PTX 中 float literals 的写法。

根据我们的测试，其他所有类型指令都可以在不做任何修改的情况下，仅从 dump 的 SASS 中 re-assemble 出完全一致的 code。

**NOTE**: 嗯……在更全面的测试后，我们发现还有一些神秘指令无法恢复，例如 Turing 上的 `B2R`、Ampere 上的 `LDG/STG`。看起来某些 modifiers 没有显示在汇编文本里，还有些指令甚至不会出现在 SASS 汇编里……任何无法从汇编文本中完全恢复的指令（如果文本里还有的话……）都不太可能在 CuAssembler 中正常工作，除非我们自己做 disassembly。我们已向 nvidia 提交 bug report，希望他们能修复……

## Instruction Assembler Repository

CuAssembler 需要大量（且足够多样化）的输入来构建用于 instruction encoding 的所有矩阵。目前 CUDA toolkit 的每个版本都会提供一批带有 built-in kernels 的库（通常位于 CUDA 安装路径的 `bin` 目录下；windows 后缀 `.dll`，linux 后缀 `.so`）。用户可以把某个版本的 SASS dump 到文件，例如：

```
  cuobjdump -sass -arch sm_75 cublas64_11.dll > cublas64_11.sm_75.sass
```

**NOTE**: `cuobjdump` 没有“把结果保存到文件”的选项，它总是输出到 `stdout`；因此如果你想保存，需要把输出重定向到文件。


在 dump 的 SASS 文件中，你会看到很长的 kernel code 列表，例如：

```
Fatbin elf code:
================
arch = sm_75
code version = [1,7]
producer = <unknown>
host = windows
compile_size = 64bit

	code for sm_75
		Function : _Z7argtestPiS_S_
	.headerflags    @"EF_CUDA_SM75 EF_CUDA_PTX_SM(EF_CUDA_SM75)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;               /* 0x00000a00ff017624 */
                                                                                         /* 0x000fd000078e00ff */
        /*0010*/                   ULDC.64 UR36, c[0x0][0x160] ;                         /* 0x0000580000247ab9 */
                                                                                         /* 0x000fe20000000a00 */
        /*0020*/                   IADD3 R1, R1, -0x28, RZ ;                             /* 0xffffffd801017810 */
                                                                                         /* 0x000fe20007ffe0ff */
  ...
```


在 CuAssembler 中，`CuInsFeeder` 类可以读取这个 SASS 文件，并迭代地产出指令，包括 address、instruction code、instruction string、control codes。`cuobjdump` 使用的语法几乎与 `nvdisasm` 相同，但不会显示显式 labels 或 symbols，因此这种格式不仅适用于指令收集，也可以用于从 `nvdisasm` 汇编文本进行 assemble。

`CuInsParser` 会读入 instruction string 与 address，并把它解析为 instruction value vector 与 modifier set。示例代码片段：

```python
fname = r'TestData\CuTest\cudatest.sm_75.sass'
feeder = CuInsFeeder(fname)

cip = CuInsParser(arch='sm_75')

for  addr, code, asm, ctrl in feeder:
    print('0x%04x :   0x%06x   0x%028x   %s'% (addr, ctrl, code, asm))

    ins_key, ins_vals, ins_modi = cip.parse(asm, addr, code)
    print('    Ins_Key = %s'%ins_key)
    print('    Ins_Vals = %s'%str(ins_vals))
    print('    Ins_Modi = %s'%str(ins_modi))
```

输出可能类似：

```
0x0000 :   0x0007e8   0x0000078e00ff00000a00ff017624   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
    Ins_Key = IMAD_R_R_R_cAI
    Ins_Vals = [7, 1, 255, 255, 0, 40]
    Ins_Modi = ['0_IMAD', '0_MOV', '0_U32']
0x0010 :   0x0007f1   0x000000000a000000580000247ab9   ULDC.64 UR36, c[0x0][0x160] ;
    Ins_Key = ULDC_UR_cAI
    Ins_Vals = [7, 36, 0, 352]
    Ins_Modi = ['0_ULDC', '0_64']
0x0020 :   0x0007f1   0x000007ffe0ffffffffd801017810   IADD3 R1, R1, -0x28, RZ ;
    Ins_Key = IADD3_R_R_II_R
    Ins_Vals = [7, 1, 1, -40, 255]
    Ins_Modi = ['0_IADD3', '3_NegIntImme']
0x0030 :   0x000751   0x00000c1ee90000000024ff057981   LDG.E.SYS R5, [UR36] ;
    Ins_Key = LDG_R_AURI
    Ins_Vals = [7, 5, 36, 0]
    Ins_Modi = ['0_LDG', '0_E', '0_SYS']
```

CuAssembler 的 `CuInsAssembler` 类负责根据 value vector 与 modifier set 对 instruction 做 encoding。由于每个 instruction key 的 value 含义与 modifier set 不同，一个 `CuInsAssembler` 实例只处理一个 key。`CuInsAssemblerRepos` 类实例则持有所有已知 instruction keys 的 repository。给定一个 SASS 文件源，`CuInsAssemblerRepos` 可以用其中的指令构建 repository，并把结果保存到文件，以便后续使用：

```python
sassname = 'cublas64_11.sm_75.sass'
arch = 'sm_75'
feeder = CuInsFeeder(sassname, arch=arch)   # initialize a feeder
repos = CuInsAssemblerRepos(arch=arch)      # initialize an empty repos
repos.update(feeder)                        # Update the repos with instructions from feeder
repos.save2file('Repos.'+arch+'.txt')       # Save the repos to file, may be loaded back later
```

构建 repository 通常很耗时，因此 `InsAsmRepos` 目录提供了预构建的 repository（`sm_75` 覆盖较好，但 `sm_61` 与 `sm_86` 覆盖较差，并且可能存在错误处理）。`CuInsAssemblerRepos` 也提供了一些子程序，可用新 SASS 文件（甚至另一个 repository）来 update、verify、merge repository。

**NOTE**: 由于需要大量架构相关处理，`CuInsFeeder`、`CuInsParser`、`CuInsAssembler`、`CuInAssemblerRepos` 都是架构相关的。不要混用不同 SM 版本的实例。虽然一些 SM 版本很接近（例如 Maxwell 与 Pascal），但仍推荐为它们分别创建实例。

