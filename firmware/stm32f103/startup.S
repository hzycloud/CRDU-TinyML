.syntax unified
.cpu cortex-m3
.thumb

.section .isr_vector,"a",%progbits
.balign 256
.global g_pfnVectors
.type g_pfnVectors, %object
g_pfnVectors:
    .word _estack
    .word Reset_Handler
    .word Default_Handler /* NMI */
    .word Default_Handler /* HardFault */
    .word Default_Handler /* MemManage */
    .word Default_Handler /* BusFault */
    .word Default_Handler /* UsageFault */
    .word 0, 0, 0, 0
    .word Default_Handler /* SVCall */
    .word Default_Handler /* Debug monitor */
    .word 0
    .word Default_Handler /* PendSV */
    .word Default_Handler /* SysTick */
    /* RM0008 medium-density vector table: IRQ0 WWDG through IRQ42 USBWakeUp. */
    .rept 43
    .word Default_Handler
    .endr
.size g_pfnVectors, . - g_pfnVectors

.section .text.Reset_Handler,"ax",%progbits
.balign 4
.global Reset_Handler
.type Reset_Handler, %function
.thumb_func
Reset_Handler:
    cpsid i                 /* UART and timers are polled; no interrupt stack use. */
    ldr r0, =_estack
    msr msp, r0
    ldr r0, =_sidata
    ldr r1, =_sdata
    ldr r2, =_edata
1:
    cmp r1, r2
    bhs 2f
    ldr r3, [r0], #4
    str r3, [r1], #4
    b 1b
2:
    ldr r0, =_sbss
    ldr r1, =_ebss
    movs r2, #0
3:
    cmp r0, r1
    bhs 4f
    str r2, [r0], #4
    b 3b
4:
    /* No push or C call has occurred: paint the entire unused descending stack. */
    ldr r0, =_end
    ldr r1, =_estack
    ldr r2, =0xa5a5a5a5
5:
    cmp r0, r1
    bhs 6f
    str r2, [r0], #4
    b 5b
6:
    ldr r0, =0xe000ed08     /* SCB_VTOR: use this image's Flash vector table. */
    ldr r1, =g_pfnVectors
    str r1, [r0]
    dsb
    isb
    bl main
7:
    b 7b
.size Reset_Handler, . - Reset_Handler

.section .text.Default_Handler,"ax",%progbits
.balign 2
.global Default_Handler
.type Default_Handler, %function
.thumb_func
Default_Handler:
    b Default_Handler
.size Default_Handler, . - Default_Handler

.section .note.GNU-stack,"",%progbits
