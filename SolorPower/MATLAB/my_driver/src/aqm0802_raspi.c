#include <pigpio.h>
#include <unistd.h>
#include "aqm0802_raspi.h"

#define I2C_BUS 1
#define I2C_ADDR 0x3e
#define I2C_FLAGS 0
#define REG_ADDR_INST 0x00
#define REG_ADDR_DATA 0x40

uint32_T aqm0802Setup()
{
    uint32_T h;

    // Perform one-time Pigpio initialization
    if (gpioCfgInterfaces(0) != PI_INITIALISED) {
        if (gpioInitialise() < 0)
        {
            // pigpio initalization failed.
        }
    }
    
    h = i2cOpen(I2C_BUS,I2C_ADDR,I2C_FLAGS);

    i2cWriteByteData(h,REG_ADDR_INST,0x38); // Function set
    i2cWriteByteData(h,REG_ADDR_INST,0x39); // Function set
    i2cWriteByteData(h,REG_ADDR_INST,0x14); // Internal OSC frequency
    i2cWriteByteData(h,REG_ADDR_INST,0x73); // Contrast set
    i2cWriteByteData(h,REG_ADDR_INST,0x56); // Power/ICON/Contrast control
    i2cWriteByteData(h,REG_ADDR_INST,0x6c); // Follower control
    usleep(200*1000); // Sleep 200ms
    i2cWriteByteData(h,REG_ADDR_INST,0x38); // Function set
    i2cWriteByteData(h,REG_ADDR_INST,0x0c); // Display ON/OFF control
    i2cWriteByteData(h,REG_ADDR_INST,0x01); // Clear Display
    usleep(1000); // Sleep 1ms
    
    i2cWriteByteData(h,REG_ADDR_DATA,0x41);
    
    return h;
    
}

void aqm0802Release(uint32_T h)
{
    if (gpioCfgInterfaces(0)==PI_INITIALISED)
    {
        i2cClose(h);
        gpioTerminate();
    }
}

// Write a string to display
void writeLine(uint8_T* line, uint8_T idx)
{
   //
}