import("stdfaust.lib");
freq = hslider("frequency", 50,20,400,1);
sineOsc(f) = +(f/ma.SR) ~ ma.frac : *(2*ma.PI) : sin;
process = sineOsc(freq);
