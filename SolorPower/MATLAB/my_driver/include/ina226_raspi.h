#ifndef _INA226_RASPI_H_
#define _INA226_RASPI_H_
#include "rtwtypes.h"

uint32_T ina226Setup();
void ina226Release(uint32_T h);
int16_T readVoltage(uint32_T h);
int16_T readCurrent(uint32_T h);

#endif //_INA226_RASPI_H_
