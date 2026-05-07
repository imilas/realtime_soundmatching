import("stdfaust.lib");
saw_freq = hslider("saw_freq",{saw_freq},20,1000,1);
sine_freq = hslider("sine_freq",{sine_freq},20,1000,1);
sineOsc(f) = +(f/ma.SR) ~ ma.frac:*(2*ma.PI) : sin;
sawOsc(f) = +(f/ma.SR) ~ ma.frac;
process = sineOsc(sine_freq)+sawOsc(saw_freq);
