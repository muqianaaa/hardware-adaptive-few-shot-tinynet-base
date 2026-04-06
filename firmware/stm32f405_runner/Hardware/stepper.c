#include "stepper.h"

extern int PROC;
/* 当前坐标（逻辑单位） */
static float current_x = 0.0f;
static float current_y = 0.0f;

/* ---------------- 微秒延时（HAL 版本，兼容 CubeMX） ---------------- */
static void delay_us(uint32_t us)
{
    uint32_t ticks = (HAL_RCC_GetSysClockFreq() / 1000000) * us / 3;
    while (ticks--) { __NOP(); }
}

/* ---------------- GPIO 控制函数 ---------------- */
static void X_Enable(uint8_t en)
{
    HAL_GPIO_WritePin(EN1_GPIO_Port, EN1_Pin,
        en ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void Y_Enable(uint8_t en)
{
    HAL_GPIO_WritePin(EN2_GPIO_Port, EN2_Pin,
        en ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void X_SetDir(int positive)
{
    HAL_GPIO_WritePin(DIR1_GPIO_Port, DIR1_Pin,
        positive ? GPIO_PIN_SET : GPIO_PIN_RESET);

    delay_us(5); // HBS57 方向信号提前至少 5us
}

static void Y_SetDir(int positive)
{
    HAL_GPIO_WritePin(DIR2_GPIO_Port, DIR2_Pin,
        positive ? GPIO_PIN_SET : GPIO_PIN_RESET);

    delay_us(5);
}

static void X_StepOnce(void)
{
    HAL_GPIO_TogglePin(PUL1_GPIO_Port, PUL1_Pin);
    delay_us(STEP_PULSE_US);
    HAL_GPIO_TogglePin(PUL1_GPIO_Port, PUL1_Pin);
    delay_us(STEP_PULSE_US);
}

static void Y_StepOnce(void)
{
    HAL_GPIO_TogglePin(PUL2_GPIO_Port, PUL2_Pin);
    delay_us(STEP_PULSE_US);
    HAL_GPIO_TogglePin(PUL2_GPIO_Port, PUL2_Pin);
    delay_us(STEP_PULSE_US);
}

/* ---------------- 两轴同步直线插补（Bresenham） ---------------- */
static void XY_Move_Steps(int32_t sx, int32_t sy)
{
    /* 设置 X 轴方向 */
    if (sx >= 0) { X_SetDir(1); }
    else { X_SetDir(0); sx = -sx; }

    /* 设置 Y 轴方向 */
    if (sy >= 0) { Y_SetDir(1); }
    else { Y_SetDir(0); sy = -sy; }

    int32_t dx = sx;
    int32_t dy = sy;
    int32_t maxd = (dx > dy) ? dx : dy;

    int32_t acc_x = 0, acc_y = 0;

    for (int32_t i = 0; i < maxd; i++)
    {
        acc_x += dx;
        acc_y += dy;

        if (acc_x >= maxd)
        {
            X_StepOnce();
            acc_x -= maxd;
        }

        if (acc_y >= maxd)
        {
            Y_StepOnce();
            acc_y -= maxd;
        }
    }
}
/* 将浮点值转换为字符串（保留 prec 位小数），用于串口打印调试。
   这是一个轻量实现，避免依赖标准库 printf 的浮点支持（有些嵌入式库禁用浮点 printf）。
*/
static void ftoa_simple(float value, char *buf, int prec)
{
  if (!buf) return;
  char *p = buf;
  if (value < 0.0f) {
    *p++ = '-';
    value = -value;
  }
  int ipart = (int)value;
  float fracf = value - (float)ipart;
 
  char rev[20];
  int ri = 0;
  if (ipart == 0) rev[ri++] = '0';
  while (ipart > 0 && ri < (int)sizeof(rev)-1) {
    rev[ri++] = '0' + (ipart % 10);
    ipart /= 10;
  }
  while (ri > 0) *p++ = rev[--ri];

  if (prec > 0) {
    *p++ = '.';
    /* scale fractional */
    int mult = 1;
    for (int i = 0; i < prec; ++i) mult *= 10;
    int fpart = (int)(fracf * (float)mult + 0.5f);
    /* write fractional with leading zeros */
    int div = mult / 10;
    for (int i = 0; i < prec; ++i) {
      int digit = fpart / div;
      *p++ = '0' + digit;
      fpart %= div;
      div /= 10;
    }
  }
  *p = '\0';
}
/* ---------------- 对外接口：初始化 ---------------- */
void Motor_Init(void)
{
    HAL_GPIO_WritePin(POS1_GPIO_Port, POS1_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(POS2_GPIO_Port, POS2_Pin, GPIO_PIN_SET);
    /* 上电使能 */
    X_Enable(0);
    Y_Enable(0);
    
    current_x = 0;
    current_y = 0;
}

/* ---------------- 对外接口：开启自动运动 ---------------- */
void enable_stepper(uint8_t en)
{
    HAL_GPIO_WritePin(POS1_GPIO_Port, POS1_Pin, en ? GPIO_PIN_SET : GPIO_PIN_RESET);
    HAL_GPIO_WritePin(POS2_GPIO_Port, POS2_Pin, en ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/* ---------------- 对外接口：接收 AI 坐标并运动 ---------------- */
void publish_position_for_motor(float x, float y, int quadrant)
{
    char rbuf0[32], rbuf1[32];char buf[128];int n;
    /* 计算目标坐标增量 */
    float dx = x - current_x;
    float dy = y - current_y;

    ftoa_simple(current_x, rbuf0, 3);
    /* 调试输出 */
    
    n = snprintf(buf, sizeof(buf),
        "current_x:%s\r\n",
        rbuf0);
    if (PROC == 1){
    HAL_UART_Transmit(&huart1, (uint8_t*)buf, n, 1000);
    }
    /* 逻辑坐标 → 步数（四舍五入） */
    int32_t sx = lroundf(dx * X_STEPS_PER_CIC / X_CM_PER_CIC);
    int32_t sy = lroundf(dy * Y_STEPS_PER_CIC / Y_CM_PER_CIC);



    ftoa_simple(dx, rbuf0, 3);
    ftoa_simple(dy, rbuf1, 3);
    /* 调试输出 */
    n = snprintf(buf, sizeof(buf),
        "Move Q%d: dx=%s dy=%s  sx=%ld sy=%ld\r\n",
        quadrant, rbuf0, rbuf1, (long)sx, (long)sy);
    if (PROC == 1){
    HAL_UART_Transmit(&huart1, (uint8_t*)buf, n, 1000);
    }
    /* 执行同步运动 */
    XY_Move_Steps(sx, sy);
    if (PROC == 1){
    HAL_UART_Transmit(&huart1, (uint8_t*)"step over\r\n", 12, 1000);
    }
    /* 更新当前位置 */
    current_x = x;
    current_y = y;
}
//增量控制步进电机
void publish_delta_for_motor(float dx, float dy, int quadrant)
{
    float x = current_x + dx;
    float y = current_y + dy;
    publish_position_for_motor(x, y, quadrant);
}