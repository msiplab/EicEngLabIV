#ifndef _AQM0802_RASPI_H_
#define _AQM0802_RASPI_H_
#include "rtwtypes.h"

uint32_T aqm0802Setup();
void aqm0802Release(uint32_T h);
void writeLine(uint32_T h, uint8_T idx, uint8_T *line, uint8_T size);

#endif //_AQM0802_RASPI_H_
