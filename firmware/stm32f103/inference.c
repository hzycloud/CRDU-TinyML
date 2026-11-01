/* Scalar integer deployment backend, shared by host verification and MCU.
 * Two static buffers, no heap. A first-layer tile is consumed by max pooling
 * immediately, avoiding allocation of its 31,360-byte unfused activation.
 */
#include "mcu.h"
static uint8_t buffers[2][MCU_BUFFER_BYTES];

MCU_API int32_t mcu_requantize(int32_t value, int32_t mult, uint8_t shift) {
    int64_t product = (int64_t)value * mult;
    uint64_t magnitude = product < 0 ? (uint64_t)(-product) : (uint64_t)product;
    if (shift) magnitude = (magnitude + ((uint64_t)1 << (shift-1))) >> shift;
    return product < 0 ? -(int32_t)magnitude : (int32_t)magnitude;
}

static int32_t weight_at(const MCULayer *layer, uint32_t index) {
    uint8_t value = layer->weights[layer->bits == 4 ? index/2 : index];
    if (layer->bits == 8) return value < 128 ? value : (int32_t)value-256;
    value = (value >> ((index & 1)*4)) & 15;
    return value < 8 ? value : (int32_t)value-16;
}

static uint8_t point(const MCULayer *layer, const uint8_t *input, int y, int x, int oc) {
    int32_t sum = layer->bias[oc];
    int count = layer->depthwise ? 1 : layer->in_c;
    uint32_t wi = (uint32_t)oc*layer->kernel*layer->kernel*count;
    for (int ky=0;ky<layer->kernel;ky++) {
        int iy=y+ky-layer->kernel/2;
        for (int kx=0;kx<layer->kernel;kx++) {
            int ix=x+kx-layer->kernel/2;
            for (int ic=0;ic<count;ic++,wi++) {
                int c=layer->depthwise ? oc : ic;
                if (iy>=0 && iy<layer->height && ix>=0 && ix<layer->width) {
                    int32_t value=input[(iy*layer->width+ix)*layer->in_c+c]-layer->in_zero;
                    sum += value * weight_at(layer,wi);
                }
            }
        }
    }
    int32_t result=mcu_requantize(sum,layer->mult[oc],layer->shift[oc])+layer->out_zero;
    int32_t minimum=layer->relu ? layer->out_zero : 0;
    if (result<minimum) result=minimum;
    if (result>255) result=255;
    return (uint8_t)result;
}

MCU_API void mcu_infer(const uint8_t input[490], int32_t logits[12]) {
    const uint8_t *current=input;
    for (int i=0;i<7;i++) {
        const MCULayer *layer=&mcu_layers[i];
        uint8_t *output=buffers[i & 1];
        int h=layer->height/layer->pool, w=layer->width/layer->pool;
        for (int y=0;y<h;y++) for (int x=0;x<w;x++) for (int c=0;c<layer->out_c;c++) {
            uint8_t best=0;
            for (int py=0;py<layer->pool;py++) for (int px=0;px<layer->pool;px++) {
                uint8_t value=point(layer,current,y*layer->pool+py,x*layer->pool+px,c);
                if (value>best) best=value;
            }
            output[(y*w+x)*layer->out_c+c]=best;
        }
        current=output;
    }
    /* Integer global average; classifier weights are W8, logits use Q10. */
    uint8_t pooled[128];
    for (int c=0;c<128;c++) {
        int32_t sum=0;
        for (int p=0;p<24;p++) sum+=current[p*128+c];
        pooled[c]=(uint8_t)((sum+12)/24);
    }
    for (int oc=0;oc<12;oc++) {
        int32_t sum=mcu_fc_bias[oc];
        for (int c=0;c<128;c++) {
            uint8_t raw=mcu_fc_weights[oc*128+c];
            int32_t weight=raw<128 ? raw : (int32_t)raw-256;
            sum += ((int32_t)pooled[c]-mcu_fc_zero)*weight;
        }
        logits[oc]=mcu_requantize(sum,mcu_fc_mult[oc],mcu_fc_shift[oc]);
    }
}
