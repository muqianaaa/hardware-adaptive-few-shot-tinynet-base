#ifndef __APP_MOTOR_H__
#define __APP_MOTOR_H__

#include "main.h"
#include <usart.h>
#include <math.h>
#include <stdio.h>

#define PWM_X_CHANNEL   TIM_CHANNEL_1
#define PWM_Y_CHANNEL   TIM_CHANNEL_2
/* ---------------- 步进配置（可按实际机械修改） ---------------- */
#define X_STEPS_PER_CIC   3200.0f   // 每单位对应步数（细分后）
#define Y_STEPS_PER_CIC   3200.0f

#define X_CM_PER_CIC  2.9f   
#define Y_CM_PER_CIC  2.9f

#define STEP_PULSE_US      10        // 脉冲高/低电平宽度（越小越快）

#ifdef __cplusplus
extern "C" {
#endif

void Motor_Init(void);
void enable_stepper(uint8_t en);
void publish_position_for_motor(float x, float y, int quadrant);
void publish_delta_for_motor(float dx, float dy, int quadrant);
#ifdef __cplusplus
}
#endif

#endif
