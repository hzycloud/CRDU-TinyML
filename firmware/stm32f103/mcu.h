#ifndef MCU_H
#define MCU_H
#include <stdint.h>
#define MCU_INPUT_BYTES 490
#define MCU_LOGITS 12
#define MCU_BUFFER_BYTES 7680
#ifdef _WIN32
#define MCU_API __declspec(dllexport)
#else
#define MCU_API
#endif

typedef struct {
    const uint8_t *weights;
    const int32_t *bias, *mult;
    const uint8_t *shift;
    uint16_t height, width, in_c, out_c;
    uint8_t kernel, depthwise, pool, bits, relu;
    int32_t in_zero, out_zero;
} MCULayer;
extern const MCULayer mcu_layers[7];
extern const uint8_t mcu_fc_weights[1536];
extern const int32_t mcu_fc_bias[12], mcu_fc_mult[12];
extern const uint8_t mcu_fc_shift[12];
extern const int32_t mcu_fc_zero;
extern const uint32_t mcu_mask;
extern const uint8_t mcu_model_hash[32];
MCU_API int32_t mcu_requantize(int32_t value, int32_t mult, uint8_t shift);
MCU_API void mcu_infer(const uint8_t input[490], int32_t logits[12]);
#endif
