/*
 * http://abyz.me.uke/rpi/pigpio/cif.html
 */
#include <pigpio.h>
#include "ina226_raspi.h" 

#define I2C_BUS 1
#define I2C_ADDR 0x40
#define I2C_FLAGS 0
#define REG_ADDR_CNFG 0x00
#define REG_ADDR_CALB 0x05
#define REG_ADDR_VOLT 0x02
#define REG_ADDR_CURR 0x04

uint32_T ina226Setup()
{
    uint32_T h;

    // Perform one-time Pigpio initialization
    if (gpioCfgInterfaces(0) != PI_INITIALISED) {
        if (gpioInitialise() < 0)
        {
            // pigpio initalization failed.
            return 1;
        }
    }
    
    h = i2cOpen(I2C_BUS,I2C_ADDR,I2C_FLAGS);
    
    i2cWriteWordData(h,REG_ADDR_CNFG,0x2741); // Configuration
    i2cWriteWordData(h,REG_ADDR_CALB,0x000a); // Calibration
    
    return h;
} 

void ina226Release(uint32_T h)
{
    if (gpioCfgInterfaces(0)==PI_INITIALISED)
    {
        i2cClose(h);
        gpioTerminate();
    }
}

// Read a volutage value 
int16_T readVoltage(uint32_T h) 
{ 
   return (int16_T)i2cReadWordData(h,REG_ADDR_VOLT);
}

// Read a current value 
int16_T readCurrent(uint32_T h) 
{ 
   return (int16_T)i2cReadWordData(h,REG_ADDR_CURR);
}