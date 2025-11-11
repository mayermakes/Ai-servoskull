# ring12.py
import time
from pi5neo import Pi5Neo

def main():
    # Parameters
    spi_device = '/dev/spidev0.0'  # adjust if using a different SPI bus
    num_leds   = 12
    spi_speed_khz = 800  # default from README
    
    # Initialize
    neo = Pi5Neo(spi_device, num_leds, spi_speed_khz)
    
    # Clear all LEDs initially
    neo.clear_strip()
    neo.update_strip()
    
    try:
        while True:
            # Example: run a chase of red lights
            for i in range(num_leds):
                neo.clear_strip()
                neo.set_led_color(i, 255, 0, 0)  # red
                neo.update_strip()
                time.sleep(0.1)
            
            # Example: fill ring green, pause, then blue
            neo.fill_strip(0, 255, 0)  # green
            neo.update_strip()
            time.sleep(1.0)
            
            neo.fill_strip(0, 0, 255)  # blue
            neo.update_strip()
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        # On CTRL-C, clear the strip
        neo.clear_strip()
        neo.update_strip()
        print("\nExiting, strip cleared.")

if __name__ == '__main__':
    main()
