# 现有野火STM32F103C8T6：语音模型上板指南

本工程面向套餐标注的STM32F103C8T6，按64 KiB Flash、20 KiB SRAM链接。实物芯片丝印仍需核对。
本轮先复用seed42的AudioDSCNN结构和训练权重，不新增训练任务。模型为12类、输入49×10个已归一化MFCC特征。

## 当前完成范围

2026-11-01已在用户实际板卡上完成五个配置、各24次计时推理（每组8例×3轮，另有预热），全部12个整数输出与主机参考一致。报告见`results/stm32f103/STM32实际验证报告_2026-11-01.md`，原始数据见同目录`hardware/`。共86项项目测试通过。以下接线与命令用于复现。

交付文件包含C源码、独立NumPy整数参考、Windows主机验证库、Cortex-M3固件、串口采集脚本及8个工程测试样本。
编译大小、主机数值一致和真实上板测量是三种证据：只有串口返回并通过校验后，才算完成对应板端测试。
准备报告见`results/stm32f103/preparation_report.json`；每个配置的`manifest.json`列出实际编译大小与核验状态。

该整数后端会融合BatchNorm，额外量化输入和全局平均池化，并使用整数重定标；它与论文原伪量化协议不同。
配置名称只说明原始选择来源，不能声称这些配置已在新后端重新最优化。工程样本取自量化角色，不是最终测试准确率。
分类器保持W8；`candidate_w4`表示七个候选卷积为W4。每个W4权重以有符号低/高半字节打包，在乘加时解包。
返回logits为Q10有符号整数，除以1024得到浮点分数；目前没有拟合新后端的温度，也没有板端softmax。

## 配置与文件

| 文件夹 | 含义 |
|---|---|
| `results/stm32f103/seed42_mask0` | 全W8工程对照 |
| `results/stm32f103/seed42_mask34` | 原预算1.0下Task选择 |
| `results/stm32f103/seed42_mask2` | 原预算1.0下Cal选择 |
| `results/stm32f103/seed42_mask35` | 原预算1.0下ECE-min选择 |
| `results/stm32f103/seed42_mask127` | 候选卷积全W4，分类器W8 |

每次只烧录一个配置。`firmware.bin`写入0x08000000，`firmware.elf`可供调试，`firmware.map`用于核对分配。
`golden.npz`包含输入与已核对的预期整数输出。`model_data.c`包含真实权重和定点参数，不是随机占位网络。

## 1. 接线与识别

先断电接好调试线，再通过USB给板供电。先不接传感器、OLED、WiFi和W25Q64。

| DAP信号 | 照片中板卡调试口的丝印 |
|---|---|
| SWDIO | TMS（PA13） |
| SWCLK | TCK（PA14） |
| GND | GND |
| nRESET，可选 | RST |

按丝印接线，不按连接器朝向猜针序。若探针有VTref电压检测输入，接板卡3V3；若标的是3.3V供电输出，应按探针说明使用，避免与USB供电直接并联。
BOOT0应处于0/正常Flash启动位置。将DAP的USB也接到电脑。

串口优先使用照片下方标有`USB TO UART`的Type-C接口，上方`USB DEVICE`不是本工程使用的串口。
若板载串口桥尚未连通，核对TXD→PA10、RXD→PA9、GND共地；也可使用套件的3.3V USB转串口线按此接法。
在Windows设备管理器或下面命令中确认COM号：

```powershell
Get-CimInstance Win32_SerialPort | Select-Object DeviceID,Name
```

## 2. 先检测DAP，再烧录一个配置

以下命令在项目或解压包根目录执行。默认命令只连接探针、读取芯片/Flash信息并恢复运行，不写Flash：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/flash_stm32.ps1 -Mask 34
```

确认型号及容量后，执行烧录；此命令会替换板上原程序，且校验写入内容：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/flash_stm32.ps1 -Mask 34 -Program
```

出现`verified`/成功退出后，按一下RESET，确认BOOT0仍为0。OpenOCD已准备在`third_party/mcu_tools/xpack-openocd-0.12.0-7/`，不需要安装Keil。
若`unable to find CMSIS-DAP`，先检查DAP USB线、供电和设备管理器，不要改成其他芯片目标或取消容量检查。
若`target not examined`，核对TMS/TCK、GND和BOOT0，保留完整日志。个别探针USB模式或驱动需要按其型号调整。

## 3. 运行数值与计时测试

本项目虚拟环境已经准备好。替换真实COM号：

```powershell
.venv/Scripts/python.exe scripts/capture_stm32.py --port COM5 --model-dir results/stm32f103/seed42_mask34
```

如在另一台电脑使用，可先创建虚拟环境并安装`requirements-mcu.txt`；仅采集预编译固件不需要PyTorch。
脚本先通过DAP回读Flash并核对完整固件SHA-256，保存回读二进制与日志；然后读取串口身份、执行一次预热，再循环3轮×8个样本。它会校验CRC、序号、模型哈希、芯片类别、容量、Flash/static-RAM及全部12个整数输出。
芯片类别以SWD读取的DBGMCU_IDCODE、CPUID和Flash容量为准。ST ES096第2.2.2节说明F103用户软件无法可靠读取调试寄存器，因此不拿串口返回的运行态调试ID代替SWD核验，也不跳过型号检查。
任何一项不符都会报错并保留记录，不会继续生成“通过”的报告。典型成功输出包含`matched`，最终`capture.json`状态为`onboard_smoke_passed`。

输出位于`results/stm32f103/hardware/`新建的时间戳目录：

- `capture.json`：每次输出、推理周期、配置频率、静态RAM和观测栈水位。
- `serial_frames.json`：请求及返回帧原始十六进制，保留CRC复核依据。
- `flash_readback.bin`及`.log`：实际运行前的Flash回读与SWD芯片身份核验。
- `source_manifest.json`及`golden.npz`：本次采集使用的参数清单与工程测试输入快照。

固件尝试8 MHz外部晶振→72 MHz PLL；失败则回退8 MHz HSI，并在每帧报告配置频率。
毫秒由周期数除以该配置频率计算，尚未外测晶振频率。串口收发和电脑MFCC预处理不计入网络推理周期。
观测SRAM可由静态RAM加已观测栈水位计算，不把链接器预留的2 KiB直接称为实际峰值。无堆分配。

一个配置通过后，将上述命令中的Mask及文件夹同步替换为0、2、35、127，逐一烧录和采集。不同配置分别保存记录。
现阶段不计算8例样本的正式准确率、NLL或ECE，也不写功耗。完整测试集、温度角色和五种子的板端扩展在工程验证后开展。

## 4. 复现导出与编译

使用原项目PyTorch环境及`results/round2/speech/seed42`检查点、冻结决定和量化特征；不使用测试集校准范围。
编译器为Zig 0.14.1的Clang/LLVM C工具链，官方发布ZIP的SHA-256已经核对。下载依据保存在`third_party/mcu_tools/zig_download.json`。
解压包不附带大型编译器；可从记录的官方URL下载、按SHA-256校验后放入相同目录，或通过`--zig`传入路径。

```powershell
.venv/Scripts/python.exe scripts/prepare_stm32.py --zig third_party/mcu_tools/zig-x86_64-windows-0.14.1/zig.exe
.venv/Scripts/python.exe -m unittest tests.test_mcu -v
```

每个配置的`build_log.json`保存实际命令及编译输出；链接器会拒绝超过64 KiB Flash或不能留出2 KiB栈的镜像。
共享C运行库与独立NumPy整数参考逐输出比较，含自然样本、全0、全255和随机输入。这些是主机验证，不是上板通过。

## 论文证据与下一步

成功配置的时间戳目录、原始帧与Flash回读已保存并复核。后续若要报告板端预测质量，需开展独立温度角色拟合与冻结测试样本评价。
论文2026-10-28终稿保留为历史版本；本次实测报告另附可加入文章的英文段落。当前证据支持功能和资源验证，不能推导硬件加速、节能或校准改善。

## 原始技术资料

- ST STM32F103C8：[产品规格](https://www.st.com/en/microcontrollers-microprocessors/stm32f103c8.html)
- ST RM0008：[参考手册](https://www.st.com/resource/en/reference_manual/cd00171190-stm32f101-103-105-107-stm32f100-series-armbased-32bit-mcus-stmicroelectronics.pdf)
- ST ES096：[F103中容量器件勘误，第2.2.2节](https://www.st.com/resource/en/errata_sheet/cd00190234-stm32f101x8-b-stm32f102x8-b-and-stm32f103x8-b-medium-density-device-limitations-stmicroelectronics.pdf)
- 野火：[F103C8T6板卡与串口接线说明](https://doc.embedfire.com/stm32_products/must_read/zh/latest/doc/introduction_of_stm32/STM32/ebf_stm32f103c8t6_core/stm32f103c8t6_core.html)
- Zig：[官方发布索引](https://ziglang.org/download/index.json)
- xPack OpenOCD：[便携版安装说明](https://xpack.github.io/openocd/install/)
