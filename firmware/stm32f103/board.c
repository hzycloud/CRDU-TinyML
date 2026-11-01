#include <stdint.h>
#include "mcu.h"

/* STM32F103 medium-density registers: ST RM0008, RCC/GPIO/USART/debug/signature. */
#define REG32(address) (*(volatile uint32_t *)(uintptr_t)(address))
#define REG16(address) (*(volatile uint16_t *)(uintptr_t)(address))
#define RCC_CR        REG32(0x40021000u)
#define RCC_CFGR      REG32(0x40021004u)
#define RCC_APB2RSTR  REG32(0x4002100cu)
#define RCC_APB2ENR   REG32(0x40021018u)
#define FLASH_ACR     REG32(0x40022000u)
#define GPIOA_CRH     REG32(0x40010804u)
#define USART1_SR     REG32(0x40013800u)
#define USART1_DR     REG32(0x40013804u)
#define USART1_BRR    REG32(0x40013808u)
#define USART1_CR1    REG32(0x4001380cu)
#define USART1_CR2    REG32(0x40013810u)
#define USART1_CR3    REG32(0x40013814u)
#define DBGMCU_IDCODE REG32(0xe0042000u)
#define FLASH_KIB     REG16(0x1ffff7e0u)

/* Arm Cortex-M3 TRM: DEMCR.TRCENA and DWT_CTRL.CYCCNTENA. */
#define DEMCR        REG32(0xe000edfcu)
#define DWT_CTRL     REG32(0xe0001000u)
#define DWT_CYCCNT   REG32(0xe0001004u)
#define SRAM_BASE    0x20000000u
#define FLASH_BASE   0x08000000u
#define STACK_PAINT  0xa5a5a5a5u
#define CLOCK_WAIT   1000000u
#define UART_ERRORS  0x0fu /* ORE, NE, FE, PE */
#define REQUEST_BYTES (MCU_INPUT_BYTES + 12u)

extern uint8_t _end, _estack, _flash_end;
_Static_assert(MCU_INPUT_BYTES == 490, "UART request requires 490 input bytes");
_Static_assert(MCU_LOGITS == 12, "UART response requires twelve logits");

static int wait_bits(volatile uint32_t *reg, uint32_t mask, uint32_t value)
{
    uint32_t remaining = CLOCK_WAIT;
    while (remaining-- != 0u) {
        if ((*reg & mask) == value)
            return 1;
    }
    return 0;
}

static void select_hsi(void)
{
    RCC_CR |= 1u;
    if (wait_bits(&RCC_CR, 2u, 2u)) {
        RCC_CFGR &= ~3u;
        if (wait_bits(&RCC_CFGR, 12u, 0u))
            return;
    }
    /* An unusable HSI is unrecoverable; never claim a clock we did not select. */
    for (;;)
        __asm volatile ("wfi");
}

static uint32_t clock_init(void)
{
    select_hsi();
    RCC_CR &= ~((1u << 24) | (1u << 19) | (1u << 16));
    if (!wait_bits(&RCC_CR, 1u << 25, 0u) ||
        !wait_bits(&RCC_CR, 1u << 17, 0u))
        goto fallback;
    RCC_CR &= ~(1u << 18); /* Crystal oscillator, not external-clock bypass. */
    RCC_CFGR = 0u;
    RCC_CR |= 1u << 16;
    if (!wait_bits(&RCC_CR, 1u << 17, 1u << 17))
        goto fallback;

    FLASH_ACR = (1u << 4) | 2u; /* Prefetch, two wait states before raising HCLK. */
    /* HSE / 1 * 9; AHB / 1; APB1 / 2; APB2 / 1; ADC / 6 (12 MHz). */
    RCC_CFGR = (1u << 16) | (7u << 18) | (4u << 8) | (2u << 14);
    RCC_CR |= 1u << 24;
    if (!wait_bits(&RCC_CR, 1u << 25, 1u << 25))
        goto fallback;
    RCC_CFGR |= 2u;
    if (wait_bits(&RCC_CFGR, 12u, 8u))
        return 72000000u;

fallback:
    select_hsi();
    RCC_CR &= ~(1u << 24);
    (void)wait_bits(&RCC_CR, 1u << 25, 0u);
    RCC_CR &= ~(1u << 16);
    RCC_CFGR = 0u;
    FLASH_ACR = 0u;
    return 8000000u;
}

static void uart_init(uint32_t clock_hz)
{
    RCC_APB2ENR |= (1u << 2) | (1u << 14);
    (void)RCC_APB2ENR; /* Let the peripheral clock enable reach the APB bridge. */
    RCC_APB2RSTR |= 1u << 14;
    RCC_APB2RSTR &= ~(1u << 14);
    /* PA9: alternate push-pull, 50 MHz; PA10: floating input. */
    GPIOA_CRH = (GPIOA_CRH & ~0xff0u) | 0x4b0u;
    USART1_CR1 = 0u;
    USART1_CR2 = 0u;
    USART1_CR3 = 0u;
    USART1_BRR = (clock_hz + 57600u) / 115200u;
    USART1_CR1 = (1u << 13) | (1u << 3) | (1u << 2); /* 8N1, UE/TE/RE. */
}

static int uart_receive(uint32_t timeout_cycles)
{
    uint32_t start = DWT_CYCCNT;
    for (;;) {
        uint32_t status = USART1_SR;
        if ((status & ((1u << 5) | UART_ERRORS)) != 0u) {
            /* SR then DR clears receive errors, including an overrun. */
            uint8_t value = (uint8_t)USART1_DR;
            return (status & UART_ERRORS) != 0u ? -2 : (int)value;
        }
        if ((uint32_t)(DWT_CYCCNT - start) >= timeout_cycles)
            return -1;
    }
}

static void uart_send(const uint8_t *data, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        while ((USART1_SR & (1u << 7)) == 0u) {}
        USART1_DR = data[i];
    }
    while ((USART1_SR & (1u << 6)) == 0u) {}
}

static uint32_t crc32(const uint8_t *data, uint32_t count)
{
    uint32_t crc = 0xffffffffu;
    for (uint32_t i = 0; i < count; ++i) {
        crc ^= data[i];
        for (uint32_t bit = 0; bit < 8u; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return crc ^ 0xffffffffu;
}

static uint32_t read_u32(const uint8_t *data)
{
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static void write_u32(uint8_t *data, uint32_t value)
{
    for (uint32_t i = 0; i < 4u; ++i)
        data[i] = (uint8_t)(value >> (8u * i));
}

typedef struct {
    uint8_t data[REQUEST_BYTES];
    uint16_t used;
} RequestBuffer;

static void drop_bytes(RequestBuffer *rx, uint32_t count)
{
    for (uint32_t i = count; i < rx->used; ++i)
        rx->data[i - count] = rx->data[i];
    rx->used = (uint16_t)(rx->used - count);
}

static int request_magic(const uint8_t *data)
{
    return data[0] == 'M' && data[1] == 'Q' &&
           (data[2] == 'I' || data[2] == 'R') && data[3] == '1';
}

static uint32_t receive_request(uint8_t input[MCU_INPUT_BYTES], uint32_t *sequence,
                                uint32_t clock_hz)
{
    static RequestBuffer rx;
    for (;;) {
        while (rx.used >= 4u && !request_magic(rx.data))
            drop_bytes(&rx, 1u);
        if (rx.used >= 4u) {
            uint32_t count = rx.data[2] == 'I' ? 12u : REQUEST_BYTES;
            if (rx.used >= count) {
                *sequence = read_u32(rx.data + 4);
                if (read_u32(rx.data + count - 4u) != crc32(rx.data, count - 4u)) {
                    /* Keep embedded/overlapping next headers after a dropped byte. */
                    drop_bytes(&rx, 1u);
                    return 0u;
                }
                uint32_t kind = rx.data[2] == 'I' ? 1u : 2u;
                if (kind == 2u) {
                    for (uint32_t i = 0; i < MCU_INPUT_BYTES; ++i)
                        input[i] = rx.data[8u + i];
                }
                drop_bytes(&rx, count);
                return kind;
            }
        }
        /* 100 ms between bytes; a complete inference request takes about 44 ms. */
        int value = uart_receive(clock_hz / 10u);
        if (value >= 0) {
            rx.data[rx.used++] = (uint8_t)value;
        } else if (rx.used < 4u) {
            rx.used = 0u;
        } else {
            /* A timeout/error invalidates this candidate, not its buffered suffix. */
            drop_bytes(&rx, 1u);
        }
    }
}

static uint32_t observed_stack_peak(void)
{
    static uint32_t peak;
    uintptr_t address = (uintptr_t)&_end;
    uintptr_t top = (uintptr_t)&_estack;
    while (address < top && REG32(address) == STACK_PAINT)
        address += 4u;
    uint32_t observed = (uint32_t)(top - address);
    if (observed > peak)
        peak = observed;
    return peak;
}

static void send_response(uint32_t sequence, uint32_t status, uint32_t clock_hz,
                          uint32_t cycles, const int32_t logits[MCU_LOGITS])
{
    static uint8_t response[128];
    response[0] = 'M'; response[1] = 'R'; response[2] = 'S'; response[3] = '1';
    write_u32(response + 4, sequence);
    write_u32(response + 8, status);
    write_u32(response + 12, mcu_mask);
    write_u32(response + 16, clock_hz);
    write_u32(response + 20, cycles);
    write_u32(response + 24, (uint32_t)((uintptr_t)&_end - SRAM_BASE));
    write_u32(response + 28, observed_stack_peak());
    write_u32(response + 32, (uint32_t)((uintptr_t)&_flash_end - FLASH_BASE));
    write_u32(response + 36, DBGMCU_IDCODE);
    write_u32(response + 40, FLASH_KIB);
    for (uint32_t i = 0; i < 32u; ++i)
        response[44u + i] = mcu_model_hash[i];
    for (uint32_t i = 0; i < MCU_LOGITS; ++i)
        write_u32(response + 76u + 4u * i, status == 2u ? (uint32_t)logits[i] : 0u);
    write_u32(response + 124, crc32(response, 124u));
    uart_send(response, sizeof response);
}

int main(void)
{
    static uint8_t input[MCU_INPUT_BYTES];
    static int32_t logits[MCU_LOGITS];
    uint32_t clock_hz = clock_init();
    uart_init(clock_hz);
    DEMCR |= 1u << 24;
    DWT_CYCCNT = 0u;
    DWT_CTRL |= 1u;
    __asm volatile ("dsb\n\tisb" ::: "memory");

    for (;;) {
        uint32_t sequence;
        uint32_t status = receive_request(input, &sequence, clock_hz);
        uint32_t cycles = 0u;
        if (status == 2u) {
            __asm volatile ("dsb\n\tisb" ::: "memory");
            uint32_t start = DWT_CYCCNT;
            mcu_infer(input, logits);
            cycles = (uint32_t)(DWT_CYCCNT - start);
        }
        send_response(sequence, status, clock_hz, cycles, logits);
    }
}
