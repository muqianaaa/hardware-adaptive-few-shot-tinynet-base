/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Few-shot benchmark runner for STM32F405RGT6
  ******************************************************************************
  */
/* USER CODE END Header */
#include "main.h"
#include "gpio.h"
#include "tim.h"
#include "usart.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BOARD_NAME "stm32f405rgt6_000"
#define BOARD_FAMILY "stm32f405rgt6_real"
#define BOARD_SRAM_BYTES 196608U
#define BOARD_FLASH_BYTES 1048576U

#define RX_CAP 2048U
#define TX_CAP 4096U
#define STEM_IN_CH 3U
#define STEM_OUT_CH 16U
#define INPUT_HW 32U
#define NUM_CLASSES 10U

#define MAX_MAIN_ACT_BYTES (24U * 1024U)
#define MAX_HIDDEN_ACT_BYTES (96U * 32U * 32U)
#define HIDDEN_CCM_BYTES (64U * 1024U)
#define HIDDEN_MAIN_BYTES (MAX_HIDDEN_ACT_BYTES - HIDDEN_CCM_BYTES)
#define MAX_PLANE_ELEMS (INPUT_HW * INPUT_HW)
#define MAX_WEIGHT_SCRATCH 1024U

#define FIXED_FLASH_OVERHEAD 24576U
#define FIXED_STATIC_SRAM 12288U
#define FIXED_STACK_BYTES 2048U

#define WARMUP_RUNS 1U
#define ARCH_RUNS 1U
#define PROBE_RUNS 5U

typedef struct
{
  char op[8];
  uint16_t width_x100;
  uint8_t depth;
  uint8_t quant;
} BenchBlockSpec;

typedef struct
{
  char name[64];
  BenchBlockSpec blocks[5];
} BenchArchitecture;

typedef struct
{
  const char *probe_id;
  const char *op;
  uint8_t quant;
  uint16_t shape[4];
  uint64_t macs;
  uint32_t bytes;
} BenchProbeSpec;

typedef struct
{
  uint32_t in_channels;
  uint32_t out_channels;
  uint32_t in_hw;
  uint32_t out_hw;
  uint64_t macs;
  uint64_t params;
  uint32_t act_in;
  uint32_t act_out;
  uint32_t bytes_moved;
  uint32_t workspace;
  uint32_t code_footprint;
} BenchBlockMetrics;

static const uint16_t kBaseChannels[5] = {16U, 24U, 32U, 48U, 64U};
static const uint8_t kStrides[5] = {1U, 2U, 1U, 2U, 1U};

static const BenchProbeSpec kProbeSuite[9] = {
    {"std3x3_small_8b", "std3x3", 8U, {1U, 16U, 16U, 16U}, 16ULL * 16ULL * 16ULL * 16ULL * 9ULL, 16U * 16U * 16U * 2U},
    {"std3x3_medium_4b", "std3x3", 4U, {1U, 24U, 8U, 8U}, 8ULL * 8ULL * 24ULL * 24ULL * 9ULL, 24U * 8U * 8U * 2U},
    {"dw_sep_small_8b", "dw_sep", 8U, {1U, 16U, 16U, 16U}, 16ULL * 16ULL * (16ULL * 9ULL + 16ULL * 16ULL), 16U * 16U * 16U * 2U},
    {"dw_sep_medium_4b", "dw_sep", 4U, {1U, 24U, 8U, 8U}, 8ULL * 8ULL * (24ULL * 9ULL + 24ULL * 24ULL), 24U * 8U * 8U * 2U},
    {"mbconv_small_8b", "mbconv", 8U, {1U, 16U, 16U, 16U}, 16ULL * 16ULL * (16ULL * 64ULL + 64ULL * 9ULL + 64ULL * 16ULL), 16U * 16U * 16U * 3U},
    {"mbconv_medium_4b", "mbconv", 4U, {1U, 24U, 8U, 8U}, 8ULL * 8ULL * (24ULL * 96ULL + 96ULL * 9ULL + 96ULL * 24ULL), 24U * 8U * 8U * 3U},
    {"fc_8b", "fc", 8U, {1U, 128U, 1U, 1U}, 128ULL * 64ULL, 128U * 8U},
    {"move_4b", "move", 4U, {1U, 32U, 8U, 8U}, 0ULL, 32U * 8U * 8U},
    {"pool_2b_emulated", "pool", 2U, {1U, 32U, 4U, 4U}, 32ULL * 16ULL, 32U * 4U * 4U},
};

static const BenchArchitecture kReferenceSuite[3] = {
    {"ref_wide_shallow", {{"std3x3", 1250U, 1U, 8U}, {"std3x3", 1250U, 1U, 8U}, {"mbconv", 1000U, 1U, 8U}, {"std3x3", 1000U, 1U, 4U}, {"mbconv", 1000U, 1U, 4U}}},
    {"ref_deep_narrow", {{"dw_sep", 500U, 2U, 8U}, {"dw_sep", 750U, 2U, 8U}, {"dw_sep", 750U, 2U, 4U}, {"mbconv", 750U, 2U, 4U}, {"dw_sep", 750U, 2U, 4U}}},
    {"ref_mixed_precision_dw", {{"dw_sep", 750U, 2U, 8U}, {"dw_sep", 1000U, 2U, 4U}, {"mbconv", 750U, 2U, 2U}, {"dw_sep", 1000U, 1U, 2U}, {"mbconv", 1000U, 1U, 4U}}},
};

static char g_rx[RX_CAP];
static char g_tx[TX_CAP];
static int8_t g_input_rgb[STEM_IN_CH * INPUT_HW * INPUT_HW];
static int8_t g_act0[MAX_MAIN_ACT_BYTES];
static int8_t g_act1[MAX_MAIN_ACT_BYTES];
#if defined(__CC_ARM)
__attribute__((at(0x10000000), zero_init))
#endif
static int8_t g_hidden_ccm[HIDDEN_CCM_BYTES];
static int8_t g_hidden_main[HIDDEN_MAIN_BYTES];
static int8_t g_plane_tmp[MAX_PLANE_ELEMS];
static int32_t g_accum_plane[MAX_PLANE_ELEMS];
static int8_t g_weight_scratch[MAX_WEIGHT_SCRATCH];
static volatile uint32_t g_sink = 0U;

/* Legacy X-CUBE-AI compatibility symbols. */
float v0 = 0.0f;
float v1 = 0.0f;
float v2 = 0.0f;
float v3 = 0.0f;

void Data_Read_ADC(void)
{
}

void SystemClock_Config(void);

static void dwt_init(void);
static void uart_send(const char *text);
static bool uart_readline(char *buffer, uint32_t cap);
static const char *find_json_value(const char *json, const char *key);
static bool json_get_string(const char *json, const char *key, char *out, uint32_t out_size);
static uint16_t parse_width_x100(const char *text);
static int8_t *hidden_plane_ptr(uint32_t hidden_index, uint32_t plane_in);
static bool parse_arch_repr(const char *repr, BenchArchitecture *arch);
static uint8_t effective_quant(uint8_t quant);
static uint16_t make_divisible(uint32_t value, uint16_t divisor);
static uint16_t block_out_channels(uint8_t block_index, uint16_t width_x100);
static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor);
static void compute_block_metrics(const BenchArchitecture *arch, BenchBlockMetrics metrics[5]);
static uint32_t mix_seed(uint32_t value);
static void fill_buffer_pattern(int8_t *buffer, uint32_t count, uint32_t seed);
static void fill_input_rgb(void);
static int8_t quantized_weight_value(uint32_t seed, uint32_t idx, uint8_t quant);
static void fill_weight_scratch(uint32_t count, uint8_t quant, uint32_t seed);
static int8_t requantize_acc(int32_t acc, uint32_t normalizer);
static void zero_accum_plane(uint32_t elems);
static void write_accum_plane(int8_t *out_plane, uint32_t elems, uint32_t normalizer);
static void exec_depthwise_channel_plane(const int8_t *in_plane, uint32_t in_hw, uint32_t stride, uint8_t quant, uint32_t seed, int8_t *out_plane);
static void exec_std3x3_layer(const int8_t *in_ptr, uint32_t in_ch, uint32_t in_hw, int8_t *out_ptr, uint32_t out_ch, uint32_t stride, uint8_t quant, uint32_t seed);
static void exec_dw_sep_layer(const int8_t *in_ptr, uint32_t in_ch, uint32_t in_hw, int8_t *out_ptr, uint32_t out_ch, uint32_t stride, uint8_t quant, uint32_t seed);
static void exec_mbconv_layer(const int8_t *in_ptr, uint32_t in_ch, uint32_t in_hw, int8_t *out_ptr, uint32_t out_ch, uint32_t stride, uint8_t quant, uint32_t seed);
static void exec_stem(int8_t *out_ptr);
static void exec_global_pool_fc(const int8_t *in_ptr, uint32_t in_ch, uint32_t hw);
static void execute_architecture_once(const BenchArchitecture *arch);
static void execute_probe_once(const BenchProbeSpec *probe);
static uint32_t measure_arch_us(const BenchArchitecture *arch);
static uint32_t measure_probe_us(const BenchProbeSpec *probe);
static uint32_t estimate_peak_sram(const BenchArchitecture *arch, const BenchBlockMetrics metrics[5]);
static uint32_t estimate_flash(const BenchArchitecture *arch, const BenchBlockMetrics metrics[5]);
static const char *width_text(uint16_t width_x100);
static void build_arch_json(const BenchArchitecture *arch, char *buffer, size_t cap);
static void format_ms(uint32_t latency_us, char *buffer, size_t cap);
static void emit_static(void);
static void emit_probe_suite(void);
static void emit_reference_suite(void);
static void emit_measure_arch(const char *line);
static void reply_error(const char *cmd, const char *error);
static void handle_command(const char *line);

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_TIM2_Init();
  HAL_TIM_Base_Start(&htim2);
  dwt_init();
  fill_input_rgb();
  HAL_Delay(250U);
  uart_send("{\"ok\":true,\"cmd\":\"boot\",\"board\":\"stm32f405rgt6_runner\"}\n");

  while (1)
  {
    if (uart_readline(g_rx, sizeof(g_rx)))
    {
      handle_command(g_rx);
    }
  }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
static void dwt_init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0U;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void uart_send(const char *text)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)text, (uint16_t)strlen(text), HAL_MAX_DELAY);
}

static bool uart_readline(char *buffer, uint32_t cap)
{
  uint32_t len = 0U;
  while (1)
  {
    uint8_t ch = 0U;
    if (HAL_UART_Receive(&huart1, &ch, 1U, 1000U) != HAL_OK)
    {
      if (len == 0U)
      {
        continue;
      }
      return false;
    }
    if (ch == '\r')
    {
      continue;
    }
    if (ch == '\n')
    {
      buffer[len] = '\0';
      return len > 0U;
    }
    if (len + 1U < cap)
    {
      buffer[len++] = (char)ch;
    }
  }
}

static const char *find_json_value(const char *json, const char *key)
{
  static char pattern[48];
  const char *cursor = NULL;
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  cursor = strstr(json, pattern);
  if (cursor == NULL)
  {
    return NULL;
  }
  cursor = strchr(cursor + strlen(pattern), ':');
  if (cursor == NULL)
  {
    return NULL;
  }
  cursor++;
  while (*cursor == ' ' || *cursor == '\t')
  {
    cursor++;
  }
  return cursor;
}

static bool json_get_string(const char *json, const char *key, char *out, uint32_t out_size)
{
  const char *cursor = find_json_value(json, key);
  uint32_t len = 0U;
  if (cursor == NULL || *cursor != '\"')
  {
    return false;
  }
  cursor++;
  while (*cursor != '\0' && *cursor != '\"' && len + 1U < out_size)
  {
    if (*cursor == '\\' && *(cursor + 1) != '\0')
    {
      cursor++;
    }
    out[len++] = *cursor++;
  }
  out[len] = '\0';
  return *cursor == '\"';
}

static uint16_t parse_width_x100(const char *text)
{
  if (strcmp(text, "0.375") == 0)
  {
    return 375U;
  }
  if (strcmp(text, "0.5") == 0)
  {
    return 500U;
  }
  if (strcmp(text, "0.75") == 0)
  {
    return 750U;
  }
  if (strcmp(text, "1.25") == 0)
  {
    return 1250U;
  }
  if (strcmp(text, "1.5") == 0)
  {
    return 1500U;
  }
  return 1000U;
}

static int8_t *hidden_plane_ptr(uint32_t hidden_index, uint32_t plane_in)
{
  uint32_t offset = hidden_index * plane_in;
  if (offset + plane_in <= HIDDEN_CCM_BYTES)
  {
    return g_hidden_ccm + offset;
  }
  offset -= HIDDEN_CCM_BYTES;
  return g_hidden_main + offset;
}

static bool parse_arch_repr(const char *repr, BenchArchitecture *arch)
{
  char copy[256];
  char *token = NULL;
  uint32_t idx = 0U;

  strncpy(copy, repr, sizeof(copy) - 1U);
  copy[sizeof(copy) - 1U] = '\0';
  memset(arch, 0, sizeof(*arch));
  strncpy(arch->name, "candidate", sizeof(arch->name) - 1U);

  token = strtok(copy, "|");
  while (token != NULL && idx < 5U)
  {
    char op[8] = {0};
    char width_buf[16] = {0};
    unsigned depth = 1U;
    unsigned quant = 8U;
    if (sscanf(token, "%7[^:]:%15[^:]:%u:%u", op, width_buf, &depth, &quant) != 4)
    {
      return false;
    }
    strncpy(arch->blocks[idx].op, op, sizeof(arch->blocks[idx].op) - 1U);
    arch->blocks[idx].width_x100 = parse_width_x100(width_buf);
    arch->blocks[idx].depth = (uint8_t)depth;
    arch->blocks[idx].quant = (uint8_t)quant;
    idx++;
    token = strtok(NULL, "|");
  }
  return idx == 5U;
}

static uint8_t effective_quant(uint8_t quant)
{
  if (quant >= 8U)
  {
    return 8U;
  }
  if (quant >= 4U)
  {
    return 4U;
  }
  return 2U;
}

static uint16_t make_divisible(uint32_t value, uint16_t divisor)
{
  uint32_t rounded = ((value + divisor / 2U) / divisor) * divisor;
  return (uint16_t)(rounded < divisor ? divisor : rounded);
}

static uint16_t block_out_channels(uint8_t block_index, uint16_t width_x100)
{
  uint32_t raw = ((uint32_t)kBaseChannels[block_index] * (uint32_t)width_x100 + 500U) / 1000U;
  return make_divisible(raw, 4U);
}

static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor)
{
  return (value + divisor - 1U) / divisor;
}

static void compute_block_metrics(const BenchArchitecture *arch, BenchBlockMetrics metrics[5])
{
  uint32_t in_channels = STEM_OUT_CH;
  uint32_t in_hw = INPUT_HW;
  for (uint32_t i = 0U; i < 5U; ++i)
  {
    const BenchBlockSpec *block = &arch->blocks[i];
    BenchBlockMetrics *metric = &metrics[i];
    uint32_t out_channels = block_out_channels((uint8_t)i, block->width_x100);
    uint64_t total_macs = 0ULL;
    uint64_t total_params = 0ULL;
    uint64_t total_bytes = 0ULL;
    uint32_t total_workspace = 0U;
    uint32_t total_code = 0U;
    uint32_t current_in_channels = in_channels;
    uint32_t current_hw = in_hw;
    uint32_t first_act_in = in_channels * in_hw * in_hw;
    uint32_t last_act_out = 0U;

    for (uint32_t depth_idx = 0U; depth_idx < block->depth; ++depth_idx)
    {
      uint32_t stride = (depth_idx == 0U) ? kStrides[i] : 1U;
      uint32_t out_hw = ceil_div_u32(current_hw, stride);
      uint32_t act_in = current_in_channels * current_hw * current_hw;
      uint32_t act_out = out_channels * out_hw * out_hw;
      uint32_t q = effective_quant(block->quant);
      uint32_t bytes_moved = (uint32_t)(((uint64_t)(act_in + act_out) * (uint64_t)q) / 8ULL);
      uint64_t macs = 0ULL;
      uint64_t params = 0ULL;
      uint32_t workspace = 0U;
      uint32_t code = 0U;

      if (strcmp(block->op, "std3x3") == 0)
      {
        macs = (uint64_t)out_hw * (uint64_t)out_hw * (uint64_t)current_in_channels * (uint64_t)out_channels * 9ULL;
        params = (uint64_t)current_in_channels * (uint64_t)out_channels * 9ULL;
        workspace = (uint32_t)(act_out * 6ULL / 100ULL);
        code = 2048U;
      }
      else if (strcmp(block->op, "dw_sep") == 0)
      {
        macs = (uint64_t)out_hw * (uint64_t)out_hw * ((uint64_t)current_in_channels * 9ULL + (uint64_t)current_in_channels * (uint64_t)out_channels);
        params = (uint64_t)current_in_channels * 9ULL + (uint64_t)current_in_channels * (uint64_t)out_channels;
        workspace = (uint32_t)(act_out * 5ULL / 100ULL);
        code = 1792U;
      }
      else
      {
        uint32_t hidden = ((out_channels > current_in_channels) ? out_channels : current_in_channels) * 4U;
        macs = (uint64_t)out_hw * (uint64_t)out_hw * ((uint64_t)current_in_channels * (uint64_t)hidden + (uint64_t)hidden * 9ULL + (uint64_t)hidden * (uint64_t)out_channels);
        params = (uint64_t)current_in_channels * (uint64_t)hidden + (uint64_t)hidden * 9ULL + (uint64_t)hidden * (uint64_t)out_channels;
        workspace = (uint32_t)(act_out * 11ULL / 100ULL);
        code = 3072U;
      }

      total_macs += macs;
      total_params += params;
      total_bytes += bytes_moved;
      total_workspace += workspace;
      total_code += code;
      current_in_channels = out_channels;
      current_hw = out_hw;
      last_act_out = act_out;
    }

    metric->in_channels = in_channels;
    metric->out_channels = out_channels;
    metric->in_hw = in_hw;
    metric->out_hw = current_hw;
    metric->macs = total_macs;
    metric->params = total_params;
    metric->act_in = first_act_in;
    metric->act_out = last_act_out;
    metric->bytes_moved = (uint32_t)total_bytes;
    metric->workspace = total_workspace;
    metric->code_footprint = total_code;
    in_channels = out_channels;
    in_hw = current_hw;
  }
}

static uint32_t mix_seed(uint32_t value)
{
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

static void fill_buffer_pattern(int8_t *buffer, uint32_t count, uint32_t seed)
{
  for (uint32_t i = 0U; i < count; ++i)
  {
    uint32_t state = mix_seed(seed + i * 2654435761U);
    buffer[i] = (int8_t)((int32_t)(state % 17U) - 8);
  }
}

static void fill_input_rgb(void)
{
  fill_buffer_pattern(g_input_rgb, (uint32_t)(STEM_IN_CH * INPUT_HW * INPUT_HW), 0x13579BDFU);
}

static int8_t quantized_weight_value(uint32_t seed, uint32_t idx, uint8_t quant)
{
  static const int8_t kQ8Levels[17] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  static const int8_t kQ4Levels[9] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
  static const int8_t kQ2Levels[5] = {-2, -1, 0, 1, 2};
  uint32_t state = mix_seed(seed + idx * 2246822519U);
  if (quant >= 8U)
  {
    return kQ8Levels[state % 17U];
  }
  if (quant >= 4U)
  {
    return kQ4Levels[state % 9U];
  }
  return kQ2Levels[state % 5U];
}

static void fill_weight_scratch(uint32_t count, uint8_t quant, uint32_t seed)
{
  if (count > MAX_WEIGHT_SCRATCH)
  {
    count = MAX_WEIGHT_SCRATCH;
  }
  for (uint32_t i = 0U; i < count; ++i)
  {
    g_weight_scratch[i] = quantized_weight_value(seed, i, quant);
  }
}

static int8_t requantize_acc(int32_t acc, uint32_t normalizer)
{
  if (normalizer == 0U)
  {
    normalizer = 1U;
  }
  if (acc >= 0)
  {
    acc = (acc + (int32_t)(normalizer / 2U)) / (int32_t)normalizer;
  }
  else
  {
    acc = (acc - (int32_t)(normalizer / 2U)) / (int32_t)normalizer;
  }
  if (acc > 127)
  {
    acc = 127;
  }
  else if (acc < -128)
  {
    acc = -128;
  }
  return (int8_t)acc;
}

static void zero_accum_plane(uint32_t elems)
{
  for (uint32_t i = 0U; i < elems; ++i)
  {
    g_accum_plane[i] = 0;
  }
}

static void write_accum_plane(int8_t *out_plane, uint32_t elems, uint32_t normalizer)
{
  for (uint32_t i = 0U; i < elems; ++i)
  {
    out_plane[i] = requantize_acc(g_accum_plane[i], normalizer);
  }
}

static void exec_depthwise_channel_plane(const int8_t *in_plane, uint32_t in_hw, uint32_t stride, uint8_t quant, uint32_t seed, int8_t *out_plane)
{
  uint32_t out_hw = ceil_div_u32(in_hw, stride);
  fill_weight_scratch(9U, quant, seed);
  for (uint32_t oh = 0U; oh < out_hw; ++oh)
  {
    for (uint32_t ow = 0U; ow < out_hw; ++ow)
    {
      int32_t acc = 0;
      for (uint32_t ky = 0U; ky < 3U; ++ky)
      {
        int32_t iy = (int32_t)(oh * stride + ky) - 1;
        if (iy < 0 || iy >= (int32_t)in_hw)
        {
          continue;
        }
        for (uint32_t kx = 0U; kx < 3U; ++kx)
        {
          int32_t ix = (int32_t)(ow * stride + kx) - 1;
          if (ix < 0 || ix >= (int32_t)in_hw)
          {
            continue;
          }
          acc += (int32_t)in_plane[(uint32_t)iy * in_hw + (uint32_t)ix] * (int32_t)g_weight_scratch[ky * 3U + kx];
        }
      }
      out_plane[oh * out_hw + ow] = requantize_acc(acc, 18U);
    }
  }
}

static void exec_std3x3_layer(const int8_t *in_ptr, uint32_t in_ch, uint32_t in_hw, int8_t *out_ptr, uint32_t out_ch, uint32_t stride, uint8_t quant, uint32_t seed)
{
  uint32_t out_hw = ceil_div_u32(in_hw, stride);
  uint32_t plane_in = in_hw * in_hw;
  uint32_t plane_out = out_hw * out_hw;
  uint32_t filter_elems = in_ch * 9U;
  uint32_t normalizer = filter_elems * 2U;

  for (uint32_t oc = 0U; oc < out_ch; ++oc)
  {
    int8_t *out_plane = out_ptr + oc * plane_out;
    fill_weight_scratch(filter_elems, quant, seed + oc * 131U);
    for (uint32_t oh = 0U; oh < out_hw; ++oh)
    {
      for (uint32_t ow = 0U; ow < out_hw; ++ow)
      {
        int32_t acc = 0;
        for (uint32_t ic = 0U; ic < in_ch; ++ic)
        {
          const int8_t *in_plane = in_ptr + ic * plane_in;
          uint32_t base = ic * 9U;
          for (uint32_t ky = 0U; ky < 3U; ++ky)
          {
            int32_t iy = (int32_t)(oh * stride + ky) - 1;
            if (iy < 0 || iy >= (int32_t)in_hw)
            {
              continue;
            }
            for (uint32_t kx = 0U; kx < 3U; ++kx)
            {
              int32_t ix = (int32_t)(ow * stride + kx) - 1;
              if (ix < 0 || ix >= (int32_t)in_hw)
              {
                continue;
              }
              acc += (int32_t)in_plane[(uint32_t)iy * in_hw + (uint32_t)ix] * (int32_t)g_weight_scratch[base + ky * 3U + kx];
            }
          }
        }
        out_plane[oh * out_hw + ow] = requantize_acc(acc, normalizer);
      }
    }
  }
}

static void exec_dw_sep_layer(const int8_t *in_ptr, uint32_t in_ch, uint32_t in_hw, int8_t *out_ptr, uint32_t out_ch, uint32_t stride, uint8_t quant, uint32_t seed)
{
  uint32_t plane_in = in_hw * in_hw;
  uint32_t out_hw = ceil_div_u32(in_hw, stride);
  uint32_t plane_out = out_hw * out_hw;
  uint32_t normalizer = in_ch * 2U;

  for (uint32_t oc = 0U; oc < out_ch; ++oc)
  {
    int8_t *out_plane = out_ptr + oc * plane_out;
    zero_accum_plane(plane_out);
    fill_weight_scratch(in_ch, quant, seed + 0x01010101U + oc * 167U);
    for (uint32_t ic = 0U; ic < in_ch; ++ic)
    {
      const int8_t *in_plane = in_ptr + ic * plane_in;
      exec_depthwise_channel_plane(in_plane, in_hw, stride, quant, seed + 0x10000U + ic * 193U, g_plane_tmp);
      for (uint32_t pos = 0U; pos < plane_out; ++pos)
      {
        g_accum_plane[pos] += (int32_t)g_plane_tmp[pos] * (int32_t)g_weight_scratch[ic];
      }
    }
    write_accum_plane(out_plane, plane_out, normalizer);
  }
}

static void exec_mbconv_layer(const int8_t *in_ptr, uint32_t in_ch, uint32_t in_hw, int8_t *out_ptr, uint32_t out_ch, uint32_t stride, uint8_t quant, uint32_t seed)
{
  uint32_t hidden_ch = ((out_ch > in_ch) ? out_ch : in_ch) * 4U;
  uint32_t plane_in = in_hw * in_hw;
  uint32_t out_hw = ceil_div_u32(in_hw, stride);
  uint32_t plane_out = out_hw * out_hw;

  for (uint32_t hc = 0U; hc < hidden_ch; ++hc)
  {
    int8_t *hidden_ptr = hidden_plane_ptr(hc, plane_in);
    fill_weight_scratch(in_ch, quant, seed + 0x00110011U + hc * 211U);
    for (uint32_t pos = 0U; pos < plane_in; ++pos)
    {
      int32_t acc = 0;
      for (uint32_t ic = 0U; ic < in_ch; ++ic)
      {
        acc += (int32_t)in_ptr[ic * plane_in + pos] * (int32_t)g_weight_scratch[ic];
      }
      hidden_ptr[pos] = requantize_acc(acc, in_ch * 2U);
    }
  }

  for (uint32_t oc = 0U; oc < out_ch; ++oc)
  {
    int8_t *out_plane = out_ptr + oc * plane_out;
    zero_accum_plane(plane_out);
    fill_weight_scratch(hidden_ch, quant, seed + 0x22002200U + oc * 223U);
    for (uint32_t hc = 0U; hc < hidden_ch; ++hc)
    {
      const int8_t *hidden_ptr = hidden_plane_ptr(hc, plane_in);
      exec_depthwise_channel_plane(hidden_ptr, in_hw, stride, quant, seed + 0x33003300U + hc * 227U, g_plane_tmp);
      for (uint32_t pos = 0U; pos < plane_out; ++pos)
      {
        g_accum_plane[pos] += (int32_t)g_plane_tmp[pos] * (int32_t)g_weight_scratch[hc];
      }
    }
    write_accum_plane(out_plane, plane_out, hidden_ch * 2U);
  }
}

static void exec_stem(int8_t *out_ptr)
{
  exec_std3x3_layer(g_input_rgb, STEM_IN_CH, INPUT_HW, out_ptr, STEM_OUT_CH, 1U, 8U, 0x55AA1234U);
}

static void exec_global_pool_fc(const int8_t *in_ptr, uint32_t in_ch, uint32_t hw)
{
  uint32_t plane = hw * hw;
  int8_t pooled[80];

  if (in_ch > 80U)
  {
    in_ch = 80U;
  }

  for (uint32_t ic = 0U; ic < in_ch; ++ic)
  {
    int32_t sum = 0;
    const int8_t *plane_ptr = in_ptr + ic * plane;
    for (uint32_t pos = 0U; pos < plane; ++pos)
    {
      sum += plane_ptr[pos];
    }
    pooled[ic] = requantize_acc(sum, plane);
  }

  for (uint32_t cls = 0U; cls < NUM_CLASSES; ++cls)
  {
    int32_t acc = 0;
    fill_weight_scratch(in_ch, 8U, 0x66778899U + cls * 257U);
    for (uint32_t ic = 0U; ic < in_ch; ++ic)
    {
      acc += (int32_t)pooled[ic] * (int32_t)g_weight_scratch[ic];
    }
    g_sink += (uint32_t)(requantize_acc(acc, in_ch * 2U) + 128);
  }
}

static void execute_architecture_once(const BenchArchitecture *arch)
{
  int8_t *current_ptr = g_act0;
  int8_t *next_ptr = g_act1;
  uint32_t current_ch = STEM_OUT_CH;
  uint32_t current_hw = INPUT_HW;

  exec_stem(current_ptr);

  for (uint32_t block_idx = 0U; block_idx < 5U; ++block_idx)
  {
    const BenchBlockSpec *block = &arch->blocks[block_idx];
    uint32_t out_ch = block_out_channels((uint8_t)block_idx, block->width_x100);
    for (uint32_t depth_idx = 0U; depth_idx < block->depth; ++depth_idx)
    {
      uint32_t stride = (depth_idx == 0U) ? kStrides[block_idx] : 1U;
      uint32_t layer_seed = 0x12340000U + block_idx * 0x1010U + depth_idx * 0x11U + block->quant;
      if (strcmp(block->op, "std3x3") == 0)
      {
        exec_std3x3_layer(current_ptr, current_ch, current_hw, next_ptr, out_ch, stride, block->quant, layer_seed);
      }
      else if (strcmp(block->op, "dw_sep") == 0)
      {
        exec_dw_sep_layer(current_ptr, current_ch, current_hw, next_ptr, out_ch, stride, block->quant, layer_seed);
      }
      else
      {
        exec_mbconv_layer(current_ptr, current_ch, current_hw, next_ptr, out_ch, stride, block->quant, layer_seed);
      }

      current_ch = out_ch;
      current_hw = ceil_div_u32(current_hw, stride);
      if (current_ptr == g_act0)
      {
        current_ptr = g_act1;
        next_ptr = g_act0;
      }
      else
      {
        current_ptr = g_act0;
        next_ptr = g_act1;
      }
    }
  }

  exec_global_pool_fc(current_ptr, current_ch, current_hw);
}

static void execute_probe_once(const BenchProbeSpec *probe)
{
  uint32_t in_ch = probe->shape[1];
  uint32_t in_hw = probe->shape[2];
  uint32_t count = in_ch * in_hw * in_hw;

  fill_buffer_pattern(g_act0, count, 0x44556677U + probe->quant);

  if (strcmp(probe->probe_id, "fc_8b") == 0)
  {
    int8_t out_vec[64];
    memset(out_vec, 0, sizeof(out_vec));
    for (uint32_t oc = 0U; oc < 64U; ++oc)
    {
      int32_t acc = 0;
      fill_weight_scratch(in_ch, 8U, 0xDEADBEEFU + oc * 13U);
      for (uint32_t ic = 0U; ic < in_ch; ++ic)
      {
        acc += (int32_t)g_act0[ic] * (int32_t)g_weight_scratch[ic];
      }
      out_vec[oc] = requantize_acc(acc, in_ch * 2U);
      g_sink += (uint32_t)(out_vec[oc] + 128);
    }
    return;
  }

  if (strcmp(probe->probe_id, "move_4b") == 0)
  {
    uint32_t copy_bytes = (probe->bytes < MAX_MAIN_ACT_BYTES) ? probe->bytes : MAX_MAIN_ACT_BYTES;
    memcpy(g_act1, g_act0, copy_bytes);
    for (uint32_t i = 0U; i < 32U; ++i)
    {
      g_sink += (uint32_t)(g_act1[i] + 128);
    }
    return;
  }

  if (strcmp(probe->probe_id, "pool_2b_emulated") == 0)
  {
    uint32_t out_hw = in_hw / 2U;
    for (uint32_t ch = 0U; ch < in_ch; ++ch)
    {
      const int8_t *in_plane = g_act0 + ch * in_hw * in_hw;
      int8_t *out_plane = g_act1 + ch * out_hw * out_hw;
      for (uint32_t oh = 0U; oh < out_hw; ++oh)
      {
        for (uint32_t ow = 0U; ow < out_hw; ++ow)
        {
          int32_t acc = 0;
          for (uint32_t ky = 0U; ky < 2U; ++ky)
          {
            for (uint32_t kx = 0U; kx < 2U; ++kx)
            {
              acc += in_plane[(oh * 2U + ky) * in_hw + (ow * 2U + kx)];
            }
          }
          out_plane[oh * out_hw + ow] = requantize_acc(acc, 4U);
        }
      }
    }
    g_sink += (uint32_t)(g_act1[0] + 128);
    return;
  }

  if (strcmp(probe->op, "std3x3") == 0)
  {
    exec_std3x3_layer(g_act0, in_ch, in_hw, g_act1, in_ch, 1U, probe->quant, 0xAAA00011U);
  }
  else if (strcmp(probe->op, "dw_sep") == 0)
  {
    exec_dw_sep_layer(g_act0, in_ch, in_hw, g_act1, in_ch, 1U, probe->quant, 0xBBB00022U);
  }
  else
  {
    exec_mbconv_layer(g_act0, in_ch, in_hw, g_act1, in_ch, 1U, probe->quant, 0xCCC00033U);
  }

  g_sink += (uint32_t)(g_act1[0] + 128);
}

static uint32_t measure_arch_us(const BenchArchitecture *arch)
{
  uint64_t total_cycles = 0ULL;
  for (uint32_t i = 0U; i < WARMUP_RUNS; ++i)
  {
    execute_architecture_once(arch);
  }
  for (uint32_t i = 0U; i < ARCH_RUNS; ++i)
  {
    uint32_t start = DWT->CYCCNT;
    execute_architecture_once(arch);
    total_cycles += (uint32_t)(DWT->CYCCNT - start);
  }
  return (uint32_t)((total_cycles * 1000000ULL) / ((uint64_t)SystemCoreClock * ARCH_RUNS));
}

static uint32_t measure_probe_us(const BenchProbeSpec *probe)
{
  uint64_t total_cycles = 0ULL;
  for (uint32_t i = 0U; i < WARMUP_RUNS; ++i)
  {
    execute_probe_once(probe);
  }
  for (uint32_t i = 0U; i < PROBE_RUNS; ++i)
  {
    uint32_t start = DWT->CYCCNT;
    execute_probe_once(probe);
    total_cycles += (uint32_t)(DWT->CYCCNT - start);
  }
  return (uint32_t)((total_cycles * 1000000ULL) / ((uint64_t)SystemCoreClock * PROBE_RUNS));
}

static uint32_t estimate_peak_sram(const BenchArchitecture *arch, const BenchBlockMetrics metrics[5])
{
  uint32_t peak = (STEM_IN_CH * INPUT_HW * INPUT_HW) + (STEM_OUT_CH * INPUT_HW * INPUT_HW);

  for (uint32_t i = 0U; i < 5U; ++i)
  {
    uint32_t q = effective_quant(arch->blocks[i].quant);
    uint32_t in_bytes = (uint32_t)(((uint64_t)metrics[i].act_in * (uint64_t)q + 7ULL) / 8ULL);
    uint32_t out_bytes = (uint32_t)(((uint64_t)metrics[i].act_out * (uint64_t)q + 7ULL) / 8ULL);
    uint32_t candidate = in_bytes + out_bytes + (metrics[i].out_hw * metrics[i].out_hw * 5U) + MAX_WEIGHT_SCRATCH;

    if (strcmp(arch->blocks[i].op, "mbconv") == 0)
    {
      uint32_t hidden = ((metrics[i].out_channels > metrics[i].in_channels) ? metrics[i].out_channels : metrics[i].in_channels) * 4U;
      uint32_t hidden_bytes = (uint32_t)((((uint64_t)hidden * (uint64_t)metrics[i].in_hw * (uint64_t)metrics[i].in_hw) * (uint64_t)q + 7ULL) / 8ULL);
      candidate += hidden_bytes;
    }

    if (candidate > peak)
    {
      peak = candidate;
    }
  }

  return peak + FIXED_STATIC_SRAM + FIXED_STACK_BYTES;
}

static uint32_t estimate_flash(const BenchArchitecture *arch, const BenchBlockMetrics metrics[5])
{
  uint64_t total = FIXED_FLASH_OVERHEAD;
  for (uint32_t i = 0U; i < 5U; ++i)
  {
    uint32_t q = effective_quant(arch->blocks[i].quant);
    total += ((uint64_t)metrics[i].params * (uint64_t)q) / 8ULL;
    total += metrics[i].code_footprint;
  }
  return (uint32_t)total;
}

static void format_ms(uint32_t latency_us, char *buffer, size_t cap)
{
  snprintf(
      buffer,
      cap,
      "%lu.%03lu",
      (unsigned long)(latency_us / 1000U),
      (unsigned long)(latency_us % 1000U));
}

static const char *width_text(uint16_t width_x100)
{
  switch (width_x100)
  {
  case 375U:
    return "0.375";
  case 500U:
    return "0.5";
  case 750U:
    return "0.75";
  case 1250U:
    return "1.25";
  case 1500U:
    return "1.5";
  default:
    return "1.0";
  }
}

static void build_arch_json(const BenchArchitecture *arch, char *buffer, size_t cap)
{
  size_t pos = 0U;
  pos += (size_t)snprintf(buffer + pos, cap - pos, "{\"name\":\"%s\",\"blocks\":[", arch->name);
  for (uint32_t i = 0U; i < 5U; ++i)
  {
    pos += (size_t)snprintf(
        buffer + pos,
        cap - pos,
        "%s{\"op\":\"%s\",\"width\":%s,\"depth\":%u,\"quant\":%u}",
        (i == 0U) ? "" : ",",
        arch->blocks[i].op,
        width_text(arch->blocks[i].width_x100),
        arch->blocks[i].depth,
        arch->blocks[i].quant);
  }
  (void)snprintf(buffer + pos, cap - pos, "]}");
}

static void emit_static(void)
{
  snprintf(
      g_tx,
      sizeof(g_tx),
      "{\"ok\":true,\"cmd\":\"get_static\",\"static\":{\"name\":\"%s\",\"family\":\"%s\",\"sram_bytes\":%lu,\"flash_bytes\":%lu,\"freq_mhz\":168.0,\"dsp\":1.0,\"simd\":1.0,\"cache_kb\":0.0,\"bus_width\":32.0,\"kernel_int8\":1.0,\"kernel_int4\":1.0,\"kernel_int2\":1.0,\"runtime_type\":\"cmsis_nn\"}}\n",
      BOARD_NAME,
      BOARD_FAMILY,
      (unsigned long)BOARD_SRAM_BYTES,
      (unsigned long)BOARD_FLASH_BYTES);
  uart_send(g_tx);
}

static void emit_probe_suite(void)
{
  size_t pos = 0U;
  char latency_buf[24];
  pos += (size_t)snprintf(g_tx + pos, sizeof(g_tx) - pos, "{\"ok\":true,\"cmd\":\"run_probe_suite\",\"rows\":[");
  for (uint32_t i = 0U; i < 9U; ++i)
  {
    uint32_t latency_us = measure_probe_us(&kProbeSuite[i]);
    format_ms(latency_us, latency_buf, sizeof(latency_buf));
    pos += (size_t)snprintf(
        g_tx + pos,
        sizeof(g_tx) - pos,
        "%s{\"probe_id\":\"%s\",\"op\":\"%s\",\"quant\":%u,\"input_shape\":[%u,%u,%u,%u],\"latency_ms\":%s,\"macs\":%llu,\"bytes\":%lu}",
        (i == 0U) ? "" : ",",
        kProbeSuite[i].probe_id,
        kProbeSuite[i].op,
        kProbeSuite[i].quant,
        kProbeSuite[i].shape[0],
        kProbeSuite[i].shape[1],
        kProbeSuite[i].shape[2],
        kProbeSuite[i].shape[3],
        latency_buf,
        (unsigned long long)kProbeSuite[i].macs,
        (unsigned long)kProbeSuite[i].bytes);
  }
  snprintf(g_tx + pos, sizeof(g_tx) - pos, "]}\n");
  uart_send(g_tx);
}

static void emit_reference_suite(void)
{
  size_t pos = 0U;
  char arch_json[512];
  char latency_buf[24];
  pos += (size_t)snprintf(g_tx + pos, sizeof(g_tx) - pos, "{\"ok\":true,\"cmd\":\"run_reference_suite\",\"rows\":[");
  for (uint32_t i = 0U; i < 3U; ++i)
  {
    BenchBlockMetrics metrics[5];
    uint32_t latency_us = measure_arch_us(&kReferenceSuite[i]);
    compute_block_metrics(&kReferenceSuite[i], metrics);
    build_arch_json(&kReferenceSuite[i], arch_json, sizeof(arch_json));
    format_ms(latency_us, latency_buf, sizeof(latency_buf));
    pos += (size_t)snprintf(
        g_tx + pos,
        sizeof(g_tx) - pos,
        "%s{\"name\":\"%s\",\"architecture\":%s,\"latency_ms\":%s,\"peak_sram_bytes\":%lu,\"flash_bytes\":%lu,\"accuracy\":null}",
        (i == 0U) ? "" : ",",
        kReferenceSuite[i].name,
        arch_json,
        latency_buf,
        (unsigned long)estimate_peak_sram(&kReferenceSuite[i], metrics),
        (unsigned long)estimate_flash(&kReferenceSuite[i], metrics));
  }
  snprintf(g_tx + pos, sizeof(g_tx) - pos, "]}\n");
  uart_send(g_tx);
}

static void emit_measure_arch(const char *line)
{
  BenchArchitecture arch;
  BenchBlockMetrics metrics[5];
  char arch_name[64] = {0};
  char arch_repr[256] = {0};
  char arch_json[512];
  char latency_buf[24];

  if (!json_get_string(line, "arch_name", arch_name, sizeof(arch_name)) ||
      !json_get_string(line, "arch_repr", arch_repr, sizeof(arch_repr)) ||
      !parse_arch_repr(arch_repr, &arch))
  {
    reply_error("measure_arch", "invalid_arch");
    return;
  }

  strncpy(arch.name, arch_name, sizeof(arch.name) - 1U);
  compute_block_metrics(&arch, metrics);
  build_arch_json(&arch, arch_json, sizeof(arch_json));
  format_ms(measure_arch_us(&arch), latency_buf, sizeof(latency_buf));

  snprintf(
      g_tx,
      sizeof(g_tx),
      "{\"ok\":true,\"cmd\":\"measure_arch\",\"row\":{\"device_name\":\"%s\",\"family\":\"%s\",\"arch_name\":\"%s\",\"arch_repr\":\"%s\",\"architecture_json\":%s,\"latency_ms\":%s,\"peak_sram_bytes\":%lu,\"flash_bytes\":%lu,\"accuracy\":null}}\n",
      BOARD_NAME,
      BOARD_FAMILY,
      arch_name,
      arch_repr,
      arch_json,
      latency_buf,
      (unsigned long)estimate_peak_sram(&arch, metrics),
      (unsigned long)estimate_flash(&arch, metrics));
  uart_send(g_tx);
}

static void reply_error(const char *cmd, const char *error)
{
  snprintf(g_tx, sizeof(g_tx), "{\"ok\":false,\"cmd\":\"%s\",\"error\":\"%s\"}\n", cmd, error);
  uart_send(g_tx);
}

static void handle_command(const char *line)
{
  char cmd[32] = {0};
  if (!json_get_string(line, "cmd", cmd, sizeof(cmd)))
  {
    reply_error("unknown", "missing_cmd");
    return;
  }

  if (strcmp(cmd, "ping") == 0)
  {
    uart_send("{\"ok\":true,\"cmd\":\"ping\",\"board\":\"stm32f405rgt6_runner\"}\n");
  }
  else if (strcmp(cmd, "get_static") == 0)
  {
    emit_static();
  }
  else if (strcmp(cmd, "run_probe_suite") == 0)
  {
    emit_probe_suite();
  }
  else if (strcmp(cmd, "run_reference_suite") == 0)
  {
    emit_reference_suite();
  }
  else if (strcmp(cmd, "measure_arch") == 0)
  {
    emit_measure_arch(line);
  }
  else
  {
    reply_error(cmd, "unsupported_cmd");
  }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
